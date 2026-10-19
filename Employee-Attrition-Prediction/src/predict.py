"""Input validation + prediction. Also works as a console app.

Run:  python src/predict.py
"""
import joblib
import pandas as pd

from utils import DEPARTMENTS, FEATURE_COLUMNS, MODEL_PATH


def load_model(path=MODEL_PATH):
    """Load the saved model, with friendly errors."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Trained model not found. Run:  python src/train_model.py")
    except Exception as exc:
        raise RuntimeError(f"Could not load the model file ({exc}). "
                           "Re-train it with: python src/train_model.py")


def validate_input(data: dict) -> dict:
    """Check every field; raise ValueError with a clear message if invalid."""
    missing = [c for c in FEATURE_COLUMNS if data.get(c) in (None, "")]
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")

    clean = {}
    try:
        clean["Age"] = int(float(data["Age"]))
        clean["MonthlyIncome"] = float(data["MonthlyIncome"])
        clean["YearsAtCompany"] = float(data["YearsAtCompany"])
        clean["DistanceFromHome"] = float(data["DistanceFromHome"])
        for score in ("JobSatisfaction", "WorkLifeBalance", "PerformanceRating"):
            clean[score] = int(float(data[score]))
    except (TypeError, ValueError):
        raise ValueError("Age, income, years, distance and the 1-5 scores must be numbers.")

    if not 18 <= clean["Age"] <= 70:
        raise ValueError("Age must be between 18 and 70.")
    if clean["MonthlyIncome"] <= 0:
        raise ValueError("Monthly income must be greater than 0.")
    if clean["YearsAtCompany"] < 0:
        raise ValueError("Years at company cannot be negative.")
    if clean["YearsAtCompany"] > clean["Age"] - 16:
        raise ValueError("Years at company is not possible for this age.")
    if clean["DistanceFromHome"] < 0:
        raise ValueError("Distance from home cannot be negative.")
    for score in ("JobSatisfaction", "WorkLifeBalance", "PerformanceRating"):
        if not 1 <= clean[score] <= 5:
            raise ValueError(f"{score} must be between 1 and 5.")

    # Case-insensitive match so "sales" or "SALES" also work
    dept = {d.lower(): d for d in DEPARTMENTS}.get(str(data["Department"]).strip().lower())
    if dept is None:
        raise ValueError(f"Department must be one of: {', '.join(DEPARTMENTS)}.")
    clean["Department"] = dept

    overtime = str(data["OverTime"]).strip().lower()
    if overtime not in ("yes", "no"):
        raise ValueError("OverTime must be Yes or No.")
    clean["OverTime"] = overtime.capitalize()
    return clean


def predict_attrition(data: dict, model_bundle=None) -> dict:
    """Return {'prediction': 'Yes'/'No', 'probability': float}."""
    model_bundle = model_bundle or load_model()
    row = pd.DataFrame([validate_input(data)])[FEATURE_COLUMNS]
    probability = float(model_bundle["pipeline"].predict_proba(row)[0, 1])
    return {"prediction": "Yes" if probability >= 0.5 else "No",
            "probability": probability}


def run_console():
    print("=== Employee Attrition Predictor ===")
    try:
        bundle = load_model()
    except (FileNotFoundError, RuntimeError) as err:
        print(f"Error: {err}")
        return
    while True:
        raw = {
            "Age": input("Age: ").strip(),
            "Department": input(f"Department ({'/'.join(DEPARTMENTS)}): ").strip(),
            "MonthlyIncome": input("Monthly income (Rs.): ").strip(),
            "YearsAtCompany": input("Years at company: ").strip(),
            "JobSatisfaction": input("Job satisfaction (1-5): ").strip(),
            "WorkLifeBalance": input("Work-life balance (1-5): ").strip(),
            "PerformanceRating": input("Performance rating (1-5): ").strip(),
            "OverTime": input("Overtime (Yes/No): ").strip(),
            "DistanceFromHome": input("Distance from home (km): ").strip(),
        }
        try:
            result = predict_attrition(raw, bundle)
            print(f"\nWill leave? {result['prediction']}  "
                  f"(probability of leaving: {result['probability']:.1%})\n")
        except ValueError as err:
            print(f"\nInvalid input: {err}\n")
        if input("Predict another? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    run_console()
