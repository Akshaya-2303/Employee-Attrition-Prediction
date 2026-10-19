"""Generate a realistic synthetic HR dataset (1500 employees).

Attrition probability is built from a logistic formula so that the data
contains real, learnable patterns (e.g. overtime and low satisfaction
increase attrition). A few values are blanked out on purpose so the
preprocessing step has missing values to handle.

Run:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

from utils import DATA_PATH, DEPARTMENTS, RANDOM_STATE, ensure_dirs


def generate_dataset(n_rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(21, 60, n_rows)
    department = rng.choice(DEPARTMENTS, n_rows, p=[0.25, 0.30, 0.10, 0.20, 0.15])
    years = np.clip(age - 21 - rng.integers(0, 12, n_rows), 0, 35)
    # Salary (in Rs.) grows with tenure, with department differences
    dept_bonus = pd.Series(department).map(
        {"Engineering": 20000, "Finance": 12000, "Sales": 5000,
         "Marketing": 3000, "HR": 0}).to_numpy()
    income = (25000 + years * 2500 + dept_bonus
              + rng.normal(0, 8000, n_rows)).clip(15000, 250000).round(-2)
    job_sat = rng.integers(1, 6, n_rows)
    work_life = rng.integers(1, 6, n_rows)
    performance = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.03, 0.07, 0.40, 0.35, 0.15])
    overtime = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])
    distance = rng.integers(1, 41, n_rows)

    # Hidden rule that decides who leaves (logistic model)
    score = (
        -2.2
        + 1.3 * (overtime == "Yes")
        - 0.45 * (job_sat - 3)
        - 0.30 * (work_life - 3)
        - 0.000012 * (income - 60000)
        - 0.05 * years
        + 0.025 * (distance - 20)
        + 0.5 * (department == "Sales")
        + 0.4 * (department == "HR")
        - 0.02 * (age - 35)
    )
    prob = 1 / (1 + np.exp(-score))
    attrition = np.where(rng.random(n_rows) < prob, "Yes", "No")

    df = pd.DataFrame({
        "EmployeeID": np.arange(1001, 1001 + n_rows),
        "Age": age, "Department": department, "MonthlyIncome": income,
        "YearsAtCompany": years, "JobSatisfaction": job_sat,
        "WorkLifeBalance": work_life, "PerformanceRating": performance,
        "OverTime": overtime, "DistanceFromHome": distance,
        "Attrition": attrition,
    })

    # Inject ~2% missing values into a few columns (to practise imputation)
    for col in ["MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "Department"]:
        df.loc[df.sample(frac=0.02, random_state=RANDOM_STATE).index, col] = np.nan
    return df


if __name__ == "__main__":
    ensure_dirs()
    data = generate_dataset()
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(data)} rows to {DATA_PATH}")
    print(f"Attrition rate: {(data['Attrition'] == 'Yes').mean():.1%}")
