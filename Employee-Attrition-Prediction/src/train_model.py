"""Train, tune, compare and save the attrition model.

Run:  python src/train_model.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_data, split_features_target
from utils import METRICS_PATH, MODEL_PATH, RANDOM_STATE, VISUALS_DIR, ensure_dirs


def get_candidate_models():
    """Two models + the hyper-parameters GridSearchCV will try.

    Class imbalance is handled with class_weight="balanced": the rare
    class (employees who leave) gets a higher weight during training."""
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [5, 10, None],
             "model__min_samples_leaf": [1, 5]},
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def get_feature_importance(pipeline) -> pd.Series:
    """Importance of each (encoded) feature from the final model."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:                                  # Logistic Regression: |coefficient|
        values = np.abs(model.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, model_name: str):
    top = importance.head(10)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top.values, color="teal")
    plt.title(f"Top Factors Influencing Attrition ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "4_feature_importance.png", dpi=150)
    plt.close()


def plot_model_comparison(results: pd.DataFrame):
    results.plot(kind="bar", figsize=(8, 4.5), colormap="Set2")
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "5_model_comparison.png", dpi=150)
    plt.close()


def train():
    ensure_dirs()
    X, y = split_features_target(load_data())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows = {}, {}
    for name, (model, grid) in get_candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        # Tune on ROC-AUC: it does not depend on the 0.5 threshold and suits imbalanced data
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        rows[name] = evaluate(search.best_estimator_, X_test, y_test)
        print(f"{name}: best params = {search.best_params_}")

    results = pd.DataFrame(rows).T.round(3)
    print("\nModel comparison on the test set:\n", results.to_string())

    
    best_name = results.sort_values(["ROC-AUC", "F1-score"], ascending=False).index[0]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    results.to_csv(METRICS_PATH)
    plot_model_comparison(results)
    importance = get_feature_importance(best_model)
    plot_feature_importance(importance, best_name)
    print("\nTop 5 factors:\n", importance.head(5).round(3).to_string())


    joblib.dump({"pipeline": best_model, "model_name": best_name}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
