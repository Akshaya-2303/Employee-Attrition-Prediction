"""Exploratory Data Analysis + the 3 data plots.

Run:  python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data
from utils import VISUALS_DIR, ensure_dirs

sns.set_theme(style="whitegrid")


def attrition_rates(df):
    """Print attrition % by department, overtime and job satisfaction."""
    df = df.assign(Left=(df["Attrition"] == "Yes").astype(int))
    print(f"\nOverall attrition rate: {df['Left'].mean():.1%}")
    for col in ["Department", "OverTime", "JobSatisfaction"]:
        print(f"\nAttrition rate by {col}:")
        print((df.groupby(col)["Left"].mean() * 100).round(1).sort_values(ascending=False).to_string())
    return df


def plot_attrition_by_department(df):
    rate = df.groupby("Department")["Left"].mean().mul(100).sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="viridis", legend=False)
    for bar, val in zip(ax.patches, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center")
    plt.title("Attrition Rate by Department")
    plt.ylabel("Attrition rate (%)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "1_attrition_by_department.png", dpi=150)
    plt.close()


def plot_income_vs_attrition(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition",
                palette="Set2", legend=False)
    plt.title("Monthly Income (Rs.) vs Attrition")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "2_income_vs_attrition.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    numeric = df.select_dtypes("number").drop(columns=["EmployeeID"])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (numerical features + Attrition)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "3_correlation_heatmap.png", dpi=150)
    plt.close()
    corr = numeric.corr()["Left"].drop("Left").sort_values(key=abs, ascending=False)
    print("\nCorrelation of each feature with attrition (strongest first):")
    print(corr.round(3).to_string())


def run_eda():
    ensure_dirs()
    df = attrition_rates(load_data())
    plot_attrition_by_department(df)
    plot_income_vs_attrition(df)
    plot_correlation_heatmap(df)
    print(f"\nPlots saved in {VISUALS_DIR}")


if __name__ == "__main__":
    run_eda()
