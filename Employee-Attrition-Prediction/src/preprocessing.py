"""Data loading and preprocessing.

All transformations live inside a scikit-learn ColumnTransformer. It is put
inside a Pipeline together with the model, so imputation / scaling /
encoding are learned ONLY from training data -> no data leakage.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import CATEGORICAL_FEATURES, DATA_PATH, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV and run basic cleaning (duplicates, bad target values)."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run:  python src/generate_data.py")
    df = df.drop_duplicates(subset="EmployeeID")
    df = df[df[TARGET].isin(["Yes", "No"])]          # drop rows with unusable target
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (1 = left, 0 = stayed). EmployeeID is dropped
    because an ID carries no real information."""
    X = df[FEATURE_COLUMNS]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Missing values -> imputation, numbers -> scaling, text -> one-hot."""
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # median is robust to outliers
        ("scale", StandardScaler()),                    # needed by Logistic Regression
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])
