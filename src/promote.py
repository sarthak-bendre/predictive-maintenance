"""Promotion gate: decide whether the challenger replaces the production champion.

1. Quality gate: the challenger must reach MIN_RECALL and MIN_PR_AUC on the
   held-out test set, or the step fails (exit code 1), which fails CI.
2. Champion vs. challenger: both are scored on the *same* current test set, each
   at its own tuned threshold. The challenger is promoted only if it lowers the
   business cost (10 x missed failures + false alarms); PR-AUC breaks ties.
   If the champion was trained on rows that are now in the test set, the
   comparison is biased toward the champion, which is the safe direction.
3. On promotion, the "champion" alias moves to the challenger's version and the
   model is exported to models/ for the API image.
"""
import json
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException

from src.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MIN_PR_AUC,
    MIN_RECALL,
    MODELS_DIR,
    PROCESSED_DIR,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TRACKING_URI,
)
from src.evaluate import classification_metrics
from src.features import FEATURE_COLS, build_xy
from src.modeling import SEARCH_SPACES, parse_run_name


def passes_gate(metrics: dict) -> tuple[bool, str]:
    failures = []
    if metrics["recall"] < MIN_RECALL:
        failures.append(f"recall {metrics['recall']:.3f} < {MIN_RECALL}")
    if metrics["pr_auc"] < MIN_PR_AUC:
        failures.append(f"PR-AUC {metrics['pr_auc']:.3f} < {MIN_PR_AUC}")
    return (not failures), ("; ".join(failures) or "passed")


def decide(challenger: dict, champion: dict | None) -> tuple[bool, str]:
    """Should the challenger replace the champion? Lower cost wins, PR-AUC breaks ties."""
    if champion is None:
        return True, "no champion yet"
    if challenger["cost"] < champion["cost"]:
        return True, f"cost {challenger['cost']:.0f} < champion {champion['cost']:.0f}"
    if challenger["cost"] == champion["cost"] and challenger["pr_auc"] > champion["pr_auc"]:
        return True, f"same cost, PR-AUC {challenger['pr_auc']:.3f} > {champion['pr_auc']:.3f}"
    return False, (f"cost {challenger['cost']:.0f} vs champion {champion['cost']:.0f}, "
                   f"PR-AUC {challenger['pr_auc']:.3f} vs {champion['pr_auc']:.3f}")


def load_alias(client, alias: str):
    """(version, model, threshold, run_name) for an alias, or None if unset."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except MlflowException:
        return None
    run = client.get_run(mv.run_id)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
    return mv, model, float(run.data.params["threshold"]), run.info.run_name


def pipeline_spec(model, run_name: str) -> dict:
    """Enough to rebuild this exact model without MLflow (see src/bootstrap.py)."""
    name, strategy = parse_run_name(run_name)
    params = model.get_params()
    return {"name": name, "strategy": strategy,
            "params": {k: params[k] for k in SEARCH_SPACES.get(name, {})}}


def export(mv, model, threshold: float, run_name: str, metrics: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "model": run_name,
        "pipeline": pipeline_spec(model, run_name),
        "threshold": threshold,
        "features": FEATURE_COLS,
        "mlflow_run_id": mv.run_id,
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": int(mv.version),
        "alias": CHAMPION_ALIAS,
        "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
    }, indent=2))


def main() -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    X_test, y_test = build_xy(pd.read_csv(PROCESSED_DIR / "test.csv"))

    def score(entry):
        _, model, threshold, _ = entry
        return classification_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)

    challenger = load_alias(client, CHALLENGER_ALIAS)
    if challenger is None:
        print("No challenger registered; run src.train first.")
        return 1
    champion = load_alias(client, CHAMPION_ALIAS)
    if champion is not None and champion[0].version == challenger[0].version:
        print(f"v{challenger[0].version} is already the champion.")
        return 0

    ch_metrics = score(challenger)
    ok, gate_reason = passes_gate(ch_metrics)
    champ_metrics = score(champion) if champion else None
    promote, reason = decide(ch_metrics, champ_metrics) if ok else (False, gate_reason)

    summary = {
        "challenger": {"version": int(challenger[0].version), "run": challenger[3], **ch_metrics},
        "champion": ({"version": int(champion[0].version), "run": champion[3], **champ_metrics}
                     if champion else None),
        "quality_gate": gate_reason,
        "promoted": promote,
        "reason": reason,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "promotion.json").write_text(json.dumps(summary, indent=2))

    label = f"challenger v{challenger[0].version} ({challenger[3]})"
    if not ok:
        print(f"REJECTED {label}: quality gate failed: {gate_reason}")
        return 1
    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, challenger[0].version)
        export(*challenger, ch_metrics)
        print(f"PROMOTED {label} to @{CHAMPION_ALIAS}: {reason}")
    else:
        print(f"KEPT champion v{champion[0].version}: {label} not better ({reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
