"""Reproducible introductory concrete-strength XAI experiment."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main():
    dataset = fetch_ucirepo(id=165)
    X = dataset.data.features.copy()
    y_frame = dataset.data.targets
    y = y_frame.iloc[:, 0].astype(float)
    if X.shape[1] != 8 or len(X) != 1030:
        raise ValueError(f"Unexpected dataset shape: {X.shape}")
    X = X.astype(float)
    X.columns = [
        "Cement", "Blast furnace slag", "Fly ash", "Water",
        "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age",
    ]

    print(f"Loaded {len(X)} samples, {X.shape[1]} features; missing values={int(X.isna().sum().sum())}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=-1)
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=1,
    )
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    pred = model.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "MAE_MPa": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE_MPa": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4),
        }

    results = {
        "dataset": "UCI Concrete Compressive Strength (id=165)",
        "train_samples": len(X_train), "test_samples": len(X_test),
        "cv_folds": 5,
        "cv_train_MAE_mean_MPa": round(float(-scores["test_mae"].mean()), 4),
        "cv_train_MAE_std_MPa": round(float(scores["test_mae"].std()), 4),
        "cv_train_R2_mean": round(float(scores["test_r2"].mean()), 4),
        "baseline_test": metrics(y_test, baseline_pred),
        "random_forest_test": metrics(y_test, pred),
    }
    print(json.dumps(results, indent=2))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred, alpha=0.55, s=20)
    low = min(float(y_test.min()), float(pred.min()))
    high = max(float(y_test.max()), float(pred.max()))
    plt.plot([low, high], [low, high], linestyle="--", color="black")
    plt.xlabel("Observed strength (MPa)")
    plt.ylabel("Predicted strength (MPa)")
    plt.title("Random Forest: independent test set")
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Tree SHAP: explain a manageable sample of held-out test observations.
    explain_X = X_test.iloc[:100].copy()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(explain_X)
    shap.plots.beeswarm(explanation, show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    shap.plots.waterfall(explanation[0], show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(OUT / "shap_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics to {ROOT / 'reports' / 'metrics.json'}")
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
