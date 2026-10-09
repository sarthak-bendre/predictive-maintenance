"""SHAP explanations for the selected model: global drivers and single predictions.

The pipeline scales features before the classifier, so SHAP runs on the scaled
matrix but plots and reasons show the original sensor values.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features import build_xy


def make_explainer(model):
    """Build the SHAP explainer once; reuse it for every prediction."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return shap.TreeExplainer(clf)
    # Features are standardised, so an all-zeros row is the training mean.
    n_features = model.named_steps["scaler"].n_features_in_
    return shap.LinearExplainer(clf, shap.maskers.Independent(np.zeros((1, n_features))))


def shap_explanation(model, X: pd.DataFrame, explainer=None) -> shap.Explanation:
    """SHAP contributions toward 'failure' for X, reported on the original sensor scale."""
    explainer = explainer or make_explainer(model)
    exp = explainer(model.named_steps["scaler"].transform(X))
    values, base = exp.values, exp.base_values
    if values.ndim == 3:  # (rows, features, classes) -> failure class
        values, base = values[:, :, 1], base[:, 1]
    return shap.Explanation(values=values, base_values=base,
                            data=X.values, feature_names=list(X.columns))


def top_contributions(exp_row: shap.Explanation, k: int = 3) -> list[dict]:
    """The k features that pushed this row hardest toward failure."""
    order = np.argsort(-exp_row.values)[:k]
    return [
        {"feature": exp_row.feature_names[i],
         "value": round(float(exp_row.data[i]), 2),
         "contribution": round(float(exp_row.values[i]), 4)}
        for i in order if exp_row.values[i] > 0
    ]


def top_reasons(exp_row: shap.Explanation, k: int = 3) -> list[str]:
    """Human-readable version of top_contributions."""
    return [f"{c['feature']} = {c['value']:g} (+{c['contribution']:.3f})"
            for c in top_contributions(exp_row, k)]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
