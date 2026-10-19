"""Shared constants and helper functions used across the project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "employee_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "model_comparison.csv"
VISUALS_DIR = ROOT_DIR / "visuals"

RANDOM_STATE = 42

DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]

NUMERIC_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "PerformanceRating", "DistanceFromHome",
]
CATEGORICAL_FEATURES = ["Department", "OverTime"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Attrition"


def ensure_dirs():
    """Create output folders if they do not exist yet."""
    for folder in (DATA_PATH.parent, MODEL_PATH.parent, VISUALS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
