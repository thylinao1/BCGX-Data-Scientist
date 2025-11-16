# Customer Churn Prediction: Cost-Optimized Classification with Survival Analysis

Churn prediction model improving recall from 7% to 94% via cost-sensitive threshold optimization, generating £13.7M savings. Applied SMOTE, Cox survival analysis, and expected value framework. Rejected price sensitivity hypothesis—campaign origin and margins drive churn, not price.

---

## Context

Completed as part of BCG X Data Science Virtual Experience. Task: investigate whether price sensitivity drives customer churn for PowerCo, a major utility provider. Extended the baseline model with advanced quantitative techniques.

---

## Key Results

| Metric | Baseline | Final Model |
|--------|----------|-------------|
| Recall | 7.4% | **94.0%** |
| Cost Savings | — | **£13.7M/year** |
| AUC-ROC | 0.69 | 0.70 |

**Primary Finding:** Price is NOT the main churn driver.

---

## Technical Approach

### 1. Class Imbalance Handling
- Compared SMOTE, ADASYN, SMOTETomek, class weighting
- SMOTE selected: best precision-recall balance

### 2. Cost-Sensitive Threshold Optimization
```
Cost = FN × £50k + FP × £500 - TP × £10k
Optimal threshold: 0.05 (not default 0.5)
```

### 3. Expected Value Framework
- Break-even at 20% retention rate
- £4.0M profit at 50% retention rate

### 4. Survival Analysis
- **Cox PH Model:** Gas customers have 10% lower hazard (p=0.04)
- **Kaplan-Meier:** Clear survival differences by segment
- Net margin statistically significant (p<0.005)

### 5. Model Comparison
- Random Forest: AUC 0.70 ✓
- XGBoost: AUC 0.70
- Logistic Regression: AUC 0.60

---

## Ablation Study

| Stage | Recall | Net Cost |
|-------|--------|----------|
| Baseline | 7.4% | £12.9M |
| + SMOTE | 13.0% | £12.0M |
| + Threshold Opt. | **94.0%** | **-£0.7M** |

Threshold optimization provided **7x more impact** than resampling.

---

## Repository Structure

```
├── powerco_eda.ipynb                 # EDA: 14,606 customers, 26 features
├── powerco_feature_engineering.ipynb # 58 engineered features, log transforms
└── powerco_random_forest_model.ipynb # Full modeling pipeline
```

---

## Technologies

Python, scikit-learn, XGBoost, imbalanced-learn, lifelines (survival analysis), pandas, matplotlib

---

## Business Recommendations

1. **Don't compete on price** — margins and service quality matter more
2. **Cross-sell gas services** — 10% hazard reduction
3. **Use 0.05 threshold** — catches 94% of churners
4. **Monitor retention rate** — profitable only if >20% success rate

---

## Key Insights

- ML metrics misleading without economic cost consideration
- Threshold optimization > model tuning for business value
- Survival analysis richer than binary classification
- Campaign acquisition source predicts churn better than price

---

*BCG X Virtual Experience, extended with SMOTE, Cox PH survival analysis, cost optimization, and expected value framework.*
