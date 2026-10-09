"""Model and data-split definitions shared by training, promotion and bootstrap.

Kept free of MLflow so the dashboard can rebuild the champion without it.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, RANDOM_STATE, TARGET, TEST_SIZE
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
def prepare_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the raw data and write the (deterministic) train/test split."""
    train_df, test_df = split(validate(load_raw()))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return train_df, test_df


def parse_run_name(run_name: str) -> tuple[str, str]:
    """"random_forest__class_weight__tuned" -> ("random_forest", "class_weight")."""
    name, strategy = run_name.split("__")[:2]
    return name, strategy
