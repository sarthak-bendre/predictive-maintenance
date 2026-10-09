"""Train and compare models, log every run to MLflow, register the best one.

Candidates: {Logistic Regression, Random Forest, XGBoost} x {class weights, SMOTE}.
The winner is registered as the "challenger"; src/promote.py decides whether it
replaces the production "champion". Random Forest and XGBoost (class weights) also get a randomized hyperparameter
search. Selection and threshold tuning use 5-fold out-of-fold predictions on the
training set only; the test set is touched once, for the final report.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.config import (
    CHALLENGER_ALIAS,
    MLFLOW_EXPERIMENT,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TRACKING_URI,
)
from src.evaluate import best_threshold, classification_metrics
from src.features import build_xy
from src.modeling import SEARCH_SPACES, make_model, prepare_split


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
    train_df, test_df = prepare_split()

    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(TRACKING_URI)
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
    _, threshold, run_id, model_uri = fitted[best_run]
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "candidate", best_run)
    print(f"Best: {best_run} -> {REGISTERED_MODEL_NAME} v{version.version} "
          f"@{CHALLENGER_ALIAS} (run src.promote to gate it into production)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-only", action="store_true",
                        help="write data/processed/{train,test}.csv and stop")
    if parser.parse_args().split_only:
        prepare_split()
    else:
        np.random.seed(RANDOM_STATE)
        main()
