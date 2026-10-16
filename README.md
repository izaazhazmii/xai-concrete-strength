# Explainable AI for Concrete Compressive Strength

**Python | scikit-learn | SHAP | Civil Engineering**

## Engineering question
Can a regression model predict concrete compressive strength from mixture composition and curing age? Which inputs contribute most to individual model predictions?

## Dataset
[UCI Concrete Compressive Strength (Dataset 165)](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength): 1,030 observations, eight numeric features and compressive strength (MPa) as the response. Original dataset and associated usage terms are provided by UCI. Cite: Yeh, I.-C. (1998), *Modeling of strength of high-performance concrete using artificial neural networks*, Cement and Concrete Research 28(12), 1797–1808.

The script downloads data from UCI on first run. No employer or confidential data is required.

## Experiment design
1. Hold out **20% of observations for an independent test** with `random_state=42`.
2. Perform shuffled **5-fold cross-validation on the training 80% only** to estimate model stability. This is an introductory random holdout exercise, not external-site or prospective validation.
3. Train a mean-prediction baseline and a fixed-configuration Random Forest regressor on the training portion.
4. Evaluate **once** on the held-out test: MAE, RMSE, R². No tuning or selection is done against test results.
5. Explain held-out predictions with Tree SHAP, reporting a global beeswarm and one local waterfall plot.

SHAP describes **model attribution**, not experimental causality. The data are observational mix records, and this model should not be used to certify structural concrete or replace laboratory testing.

## Run locally
Requirements: Python 3.10+ and an internet connection to download the public dataset.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Outputs
After running, inspect:
- `reports/metrics.json` — model-vs-baseline comparison and training CV summary
- `reports/figures/actual_vs_predicted.png` — prediction quality on held-out test set
- `reports/figures/shap_beeswarm.png` — global feature attribution on a held-out sample
- `reports/figures/shap_waterfall.png` — explanation for one held-out sample

**Results are intentionally not claimed in advance.** Run the code and report the metrics you actually obtain. Plot PNGs are excluded by `.gitignore` until you choose to publish verified figures; force-add only verified figures or remove the relevant ignore rule.

## Interpretation checklist
- Which feature has the largest mean absolute SHAP attribution?
- How does the sign of its SHAP contributions vary across samples?
- How accurate is the model compared with the mean-prediction baseline?
- What limitations arise from correlations between mixture ingredients?
- How might an unseen concrete production site differ from a random holdout?

## Future improvements
Try a linear regression baseline, repeated cross-validation, an independent-source evaluation and SHAP dependence plots. Keep the independent test set untouched when comparing model configurations.
