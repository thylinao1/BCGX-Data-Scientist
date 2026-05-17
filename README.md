# PowerCo Customer Churn: Cost-Sensitive Classification with Survival Analysis

Churn modelling on the PowerCo SME dataset (14,606 customers, 9.7% churn
prevalence). The pipeline combines a SMOTE-balanced Random Forest with a
cost-sensitive operating threshold and a Cox proportional-hazards survival
model.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Contents

- [Approach](#approach)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Installation and data](#installation-and-data)
- [Reproducing the analysis](#reproducing-the-analysis)
- [Limitations](#limitations)

## Approach

The task is framed as a decision problem rather than a pure classification
problem. The model predicts churn probabilities, and the operating threshold
is selected to minimise expected business cost on a held-out validation
fold:

```
expected_cost = FN * CLV
              + FP * campaign_cost
              + TP * (campaign_cost - CLV * retention_rate)
```

The three inputs (CLV, campaign_cost, retention_rate) drive every per-cell
cost so the threshold-selection logic and the expected-value sensitivity
analysis use a single set of assumptions.

The pipeline is built so the test fold is touched exactly once, at the end,
to report final metrics. Threshold tuning and any hyper-parameter choices
happen on training and validation folds only.

The main components are:

1. Three-way 60 / 20 / 20 stratified split into train, validation and test.
2. SMOTE oversampling on the training fold inside an `imblearn.Pipeline`
   so synthetic samples are refit per fold during cross-validation.
3. Cost-sensitive threshold optimisation on the validation fold.
4. 5-fold stratified cross-validation reporting the cost-optimal threshold
   as mean and standard deviation across folds.
5. Sensitivity analysis on the three cost levers (CLV, campaign cost,
   retention rate).
6. Cox proportional-hazards model with log-transformed heavy-tailed
   covariates, concordance index, and a Schoenfeld residual test of the
   proportional-hazards assumption.
7. Permutation feature importance on the test fold using the frozen
   pipeline.

## Headline results

All numbers are computed on a held-out test fold of ~2,921 customers, at the
threshold selected on a disjoint validation fold.

| Quantity | Value |
|----------|-------|
| Cost-optimal threshold (validation) | ~0.05 |
| Test-fold recall at cost-optimal threshold | ~0.9 |
| Test-fold cost reduction vs. RF baseline at t = 0.5 | ~£13M (test fold, single snapshot) |
| Threshold-optimisation gain divided by SMOTE-only gain | ~13x |
| Cox PH hazard ratio for `has_gas` | ~0.90, p ~ 0.04 |
| Cox PH concordance index | ~0.56 |

Re-run the modelling notebook to regenerate the exact values for your random
seed and library versions.

The Cox model is fit on `[tenure, churn, log1p(cons_12m), log1p(net_margin),
var_year_price_off_peak, has_gas]`. The statistically significant covariate
is `has_gas`: dual-fuel customers show a ~10% lower churn hazard
(HR ~ 0.90, p ~ 0.04). Campaign-origin features are not part of the Cox
model; that signal is reported separately by the classifier's permutation
importance.

## Repository layout

```
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- LICENSE
|-- data/
|   `-- README.md
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_feature_engineering.ipynb
|   `-- 03_modelling.ipynb
|-- src/
|   |-- data_loading.py
|   |-- features.py
|   |-- evaluation.py
|   |-- model.py
|   `-- survival.py
|-- tests/
|   |-- test_data_loading.py
|   |-- test_evaluation.py
|   |-- test_features.py
|   `-- test_model.py
`-- .github/workflows/ci.yml
```

## Installation and data

```bash
git clone https://github.com/thylinao1/BCGX-Data-Scientist.git
cd BCGX-Data-Scientist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/
```

The PowerCo data is not committed to the repository. `data/README.md`
lists the files required and the notebook that produces each intermediate
artefact.

## Reproducing the analysis

Run the notebooks in order:

1. `notebooks/01_eda.ipynb` produces `data/clean_data_after_eda.csv`.
2. `notebooks/02_feature_engineering.ipynb` produces
   `data/data_for_predictions.csv`.
3. `notebooks/03_modelling.ipynb` runs the full classification pipeline,
   the cost-sensitivity analysis, the survival analysis, and the test-fold
   evaluation.

The shared logic lives under `src/` and is covered by 19 pytest unit
tests. CI runs on every push and pull request against Python 3.10 and 3.11.

## Limitations

1. The dataset is a single 2015 snapshot, so the test-fold cost reduction
   is a one-shot estimate. It is not an annualised figure and there is no
   out-of-time validation.
2. Cost parameters (CLV £50k, campaign £500, retention rate 0.3) are
   assumed values taken from the original task description. The
   sensitivity analysis in the modelling notebook varies all three.
3. SMOTE inflates the training class prevalence; the test fold remains at
   the natural prevalence (~9.7%). Probability calibration after SMOTE is
   a known issue and would need isotonic or Platt scaling before any
   production use.
4. The dataset contains no customer-service interaction data, no
   competitor pricing, and no contract-change history. Conclusions about
   price sensitivity are limited to absolute price levels in this
   snapshot.
5. The Cox concordance index is ~0.56, which is only marginally above
   chance. The `has_gas` finding is real and statistically significant,
   but the Cox model is treated as a complement to the classifier rather
   than a standalone driver of retention strategy.

## License

MIT. See [LICENSE](LICENSE).
