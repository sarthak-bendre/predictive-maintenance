"""SHAP explanations for the selected model: global drivers and single predictions.

The pipeline scales features before the classifier, so SHAP runs on the scaled
matrix but plots and reasons show the original sensor values.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features import build_xy


def shap_explanation(model, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds / probability contribution toward 'failure') for X."""
    clf = model.named_steps["clf"]
    X_scaled = model.named_steps["scaler"].transform(X)
    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, X_scaled)
    exp = explainer(X_scaled)
    values, base = exp.values, exp.base_values
    if values.ndim == 3:  # (rows, features, classes) -> failure class
        values, base = values[:, :, 1], base[:, 1]
    return shap.Explanation(values=values, base_values=base,
                            data=X.values, feature_names=list(X.columns))


def top_reasons(exp_row: shap.Explanation, k: int = 3) -> list[str]:
    """Human-readable drivers that pushed this row toward failure."""
    order = np.argsort(-exp_row.values)[:k]
    return [
        f"{exp_row.feature_names[i]} = {exp_row.data[i]:.1f} (+{exp_row.values[i]:.3f})"
        for i in order if exp_row.values[i] > 0
    ]


def main() -> None:
    model = joblib.load(MODELS_DIR / "model.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    X_test, y_test = build_xy(pd.read_csv(PROCESSED_DIR / "test.csv"))
    exp = shap_explanation(model, X_test)
    REPORTS_DIR.mkdir(exist_ok=True)

    shap.plots.beeswarm(exp, max_display=11, show=False)
    plt.title(f"What drives predicted failure ({meta['model']})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    shap.plots.bar(exp, max_display=11, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150)
    plt.close()

    # Local explanation: the highest-risk machine that really failed.
    proba = model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(np.where(y_test == 1, proba, -1)))
    shap.plots.waterfall(exp[idx], max_display=8, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_example_failure.png", dpi=150)
    plt.close()

    print(f"Test row {idx}: p(failure) = {proba[idx]:.2f}, flagged because:")
    for reason in top_reasons(exp[idx]):
        print("  -", reason)
    mean_abs = pd.Series(np.abs(exp.values).mean(0), index=exp.feature_names)
    print("\nMean |SHAP| (global importance):")
    print(mean_abs.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
