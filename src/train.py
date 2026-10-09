"""Train and compare models, log every run to MLflow, register the best one.

Candidates: {Logistic Regression, Random Forest, XGBoost} x {class weights, SMOTE}.
Selection and threshold tuning use 5-fold out-of-fold predictions on the
training set only; the test set is touched once, for the final report.
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    ROOT,
    TARGET,
    TEST_SIZE,
)
from src.evaluate import best_threshold, classification_metrics
from src.features import FEATURE_COLS, build_xy
from src.ingest import load_raw
from src.validate import validate


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_model(name: str, strategy: str, pos_weight: float):
    """Build an (imblearn) pipeline. Scaling and SMOTE sit inside the pipeline,
    so during CV they are fitted on the training folds only."""
    weighted = strategy == "class_weight"
    if name == "logreg":
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced" if weighted else None
        )
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif name == "xgboost":
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight if weighted else 1.0,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(name)

    steps = [("scaler", StandardScaler())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def main() -> None:
    df = validate(load_raw())
    train_df, test_df = split(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows, fitted = [], {}
    for name in ["logreg", "random_forest", "xgboost"]:
        for strategy in ["class_weight", "smote"]:
            run_name = f"{name}__{strategy}"
            print(f"Training {run_name}")
            with mlflow.start_run(run_name=run_name) as run:
                model = make_model(name, strategy, pos_weight)
                oof = cross_val_predict(
                    model, X_train, y_train, cv=cv, method="predict_proba"
                )[:, 1]
                threshold = best_threshold(y_train, oof)
                cv_metrics = classification_metrics(y_train, oof, threshold)

                model.fit(X_train, y_train)
                test_proba = model.predict_proba(X_test)[:, 1]
                test_metrics = classification_metrics(y_test, test_proba, threshold)

                mlflow.log_params(
                    {"model": name, "imbalance": strategy, "threshold": threshold}
                )
                mlflow.log_params(
                    {f"clf__{k}": v for k, v in model.named_steps["clf"].get_params().items()
                     if isinstance(v, (int, float, str, bool)) or v is None}
                )
                mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                info = mlflow.sklearn.log_model(model, name="model",
                                                input_example=X_train.head(3),
                                                serialization_format="cloudpickle")

                fitted[run_name] = (model, threshold, run.info.run_id, info.model_uri)
                rows.append({"run": run_name, "threshold": threshold,
                             **{f"cv_{k}": cv_metrics[k] for k in ["pr_auc", "recall", "precision", "f1", "cost"]},
                             **{f"test_{k}": test_metrics[k] for k in ["pr_auc", "recall", "precision", "f1", "cost"]}})

    results = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(results.to_string(index=False))

    # Select on cross-validated PR-AUC, never on the test set.
    best_run = results.iloc[0]["run"]
    model, threshold, run_id, model_uri = fitted[best_run]
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"Best: {best_run} -> registered {REGISTERED_MODEL_NAME} v{version.version}")

    # Plain-file copy so the API image doesn't need the MLflow store.
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    best = results.iloc[0].to_dict()
    metadata = {
        "model": best_run,
        "threshold": threshold,
        "features": FEATURE_COLS,
        "mlflow_run_id": run_id,
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": int(version.version),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
