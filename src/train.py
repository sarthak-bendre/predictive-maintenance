"""Train and compare models, log every run to MLflow, register the best one.

Candidates: {Logistic Regression, Random Forest, XGBoost} x {class weights, SMOTE}.
Random Forest and XGBoost (class weights) also get a randomized hyperparameter
search. Selection and threshold tuning use 5-fold out-of-fold predictions on the
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
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


SEARCH_SPACES = {
    "random_forest": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 8, 12, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", 0.5, 0.8],
    },
    "xgboost": {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
}
SEARCH_ITER = 20


def tune(name: str, X, y, pos_weight: float) -> dict:
    """Randomized search on PR-AUC. Uses its own fold split (different seed) so
    the search and the out-of-fold evaluation don't share folds; nested CV would
    remove the remaining optimism and is noted as a limitation."""
    search = RandomizedSearchCV(
        make_model(name, "class_weight", pos_weight),
        SEARCH_SPACES[name],
        n_iter=SEARCH_ITER,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1),
        random_state=RANDOM_STATE,
        n_jobs=1,  # the estimators already parallelise
    )
    search.fit(X, y)
    print(f"  best search PR-AUC {search.best_score_:.3f}: {search.best_params_}")
    return search.best_params_


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

    candidates = [(f"{n}__{st}", n, st, {})
                  for n in ["logreg", "random_forest", "xgboost"]
                  for st in ["class_weight", "smote"]]
    for name in SEARCH_SPACES:
        print(f"Tuning {name}")
        candidates.append((f"{name}__class_weight__tuned", name, "class_weight",
                           tune(name, X_train, y_train, pos_weight)))

    rows, fitted = [], {}
    for run_name, name, strategy, params in candidates:
        print(f"Training {run_name}")
        with mlflow.start_run(run_name=run_name) as run:
            model = make_model(name, strategy, pos_weight).set_params(**params)
            oof = cross_val_predict(
                model, X_train, y_train, cv=cv, method="predict_proba"
            )[:, 1]
            threshold = best_threshold(y_train, oof)
            cv_metrics = classification_metrics(y_train, oof, threshold)

            model.fit(X_train, y_train)
            test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = classification_metrics(y_test, test_proba, threshold)

            mlflow.log_params({"model": name, "imbalance": strategy,
                               "tuned": bool(params), "threshold": threshold})
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
