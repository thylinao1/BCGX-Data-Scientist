# PowerCo Customer Churn: Cost-Sensitive Classification with Survival Analysis

Churn modelling on the PowerCo SME dataset (14,606 customers, 9.7% churn
prevalence). The pipeline combines a SMOTE-balanced Random Forest with a
cost-sensitive operating threshold, a Random Survival Forest for
time-to-churn, and a pseudo out-of-time validation by contract activation
date.

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
6. Random Survival Forest for time-to-churn, reporting a held-out
   concordance index and permutation importance. It replaces an earlier
   linear Cox model that reached only ~0.56 concordance on this data.
7. Permutation feature importance on the test fold using the frozen
   pipeline.
8. Probability calibration via `CalibratedClassifierCV` (isotonic) so the
   threshold lives on a meaningful probability scale.
9. Bootstrap 95% confidence intervals for recall, precision and the
   test-fold cost figure.
10. Pseudo out-of-time validation: customers are partitioned by contract
    activation date, the model is trained on the earlier cohort, and the
    cost matrix is re-evaluated on the later cohort to test for temporal
    drift.

## Headline results

All numbers are computed on a held-out test fold of ~2,921 customers, at the
threshold selected on a disjoint validation fold.

| Quantity | Value |
|----------|-------|
| Cost-optimal threshold (validation) | ~0.01 (on the floor of the search grid) |
| Test-fold recall at cost-optimal threshold | ~0.99 (~1.00 after calibration) |
| Test-fold cost reduction vs. RF baseline at t = 0.5 | ~£16M (test fold, single snapshot) |
| Threshold-optimisation gain divided by SMOTE-only gain | ~13x |
| Random Survival Forest concordance (held-out) | ~0.71 (the linear Cox model reached only ~0.56) |
| Out-of-time test AUC vs. random-split AUC | ~0.62 vs. ~0.67 |
| Out-of-time cost-optimal threshold | ~0.01 (unchanged from the random split) |

The cost-optimal threshold lands at the bottom of the threshold sweep, which
means the implied policy at the assumed cost parameters is to contact
essentially every customer. This is mathematically correct given
`CLV = £50k`, `campaign_cost = £500`, `retention_rate = 0.3` (the per-TP
benefit of £14.5k swamps the £500 per-FP contact cost), but it is not a
deployable policy. Section 11 of the modelling notebook prints the
operational profile (contact rate, contacts per saved customer, campaign
spend) so the headline figures cannot be read in isolation. Any real
deployment would either tighten the cost assumptions or add an explicit
contact-budget constraint.

Re-run the modelling notebook to regenerate the exact values for your random
seed and library versions.

The survival model is a Random Survival Forest fit on 15 curated covariates
(consumption, forecast, margin, price-variation and contract attributes),
with contract tenure as the duration and churn as the event. It reaches a
concordance index of ~0.71 on a held-out fold, against ~0.56 for the linear
Cox model it replaces. Permutation importance puts the electricity-margin
features (`margin_net_pow_ele`, `margin_gross_pow_ele`, `net_margin`) at the
top, consistent with the classifier's own driver ranking. The training
concordance is higher than the held-out figure, so the survival model is
treated as a complement to the classifier rather than a standalone retention
model.

The out-of-time validation partitions customers by contract activation date,
trains on the earliest ~80% of activations and evaluates on the most recent
~20%. The cost-optimal threshold stays at the grid floor on the later cohort,
so the *policy* is robust to drift. The classifier's discrimination is not:
test AUC falls from ~0.67 on a random split to ~0.62 on the out-of-time
cohort. The cost reduction figure remains large on the later cohort, but that
is driven by its higher churn prevalence (~14% versus ~10%) rather than by
better model quality, which is why the headline figure is reported on the
random-split test fold and the out-of-time result is reported separately as a
drift check.

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
|   |-- calibration.py
|   |-- uncertainty.py
|   |-- time_split.py
|   `-- survival.py
|-- tests/
|   |-- test_data_loading.py
|   |-- test_evaluation.py
|   |-- test_features.py
|   |-- test_calibration.py
|   |-- test_uncertainty.py
|   |-- test_time_split.py
|   |-- test_survival.py
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
   the cost-sensitivity analysis, the Random Survival Forest, the
   out-of-time validation, and the test-fold evaluation.

The shared logic lives under `src/` and is covered by 48 pytest unit
tests. CI runs on every push and pull request against Python 3.10 and 3.11.

## Limitations

1. The dataset is a single 2015 snapshot, so the test-fold cost reduction
   is a one-shot estimate, not an annualised figure. The out-of-time
   validation partitions customers by activation date, but every churn
   label is still observed at the same calendar moment, so it is a pseudo
   out-of-time test, not a substitute for genuine out-of-time data.
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
5. The Random Survival Forest reaches ~0.71 concordance on a held-out
   fold, but its training concordance is materially higher, so it carries
   some overfitting. It is treated as a complement to the classifier and a
   demonstration that tenure carries real signal, not a standalone
   retention model.

## License

MIT. See [LICENSE](LICENSE).
