# TestPilot AI: Rigorous Empirical Evaluation Summary

## 1. Experimental Overview

- **Evaluation Split**: Chronological Held-Out Test Builds (strictly unseen future builds).
- **Evaluated Test Records**: 1,920
- **Evaluated CI Builds**: 12
- **Ground-Truth Test Failures in Split**: 203

## 2. Test Prioritization Performance Across Strategies

| Prioritization Strategy | APFD | APFDc | Recall @ 5m (300s) | Recall @ 10m (600s) | Recall @ 20m (1200s) |
|---|---|---|---|---|---|
| **Baseline A: Random Ordering** | 0.5153 ± 0.053 | 0.5326 ± 0.064 | 68.2% | 100.0% | 100.0% |
| **Baseline B: Fastest-First** | 0.5162 | 0.7390 | 91.3% | 100.0% | 100.0% |
| **Baseline C: Historical Prioritization** | 0.6221 | 0.5930 | 67.6% | 100.0% | 100.0% |
| **ML: Logistic Regression (Value-Density)** | 0.5409 | 0.7454 | 91.2% | 100.0% | 100.0% |
| **ML: Random Forest (Value-Density)** | 0.5227 | 0.7319 | 90.9% | 100.0% | 100.0% |
| **ML: XGBoost (Value-Density)** | 0.5243 | 0.7351 | 90.6% | 100.0% | 100.0% |
| **ML: XGBoost (Pure Risk)** | 0.5592 | 0.5443 | 70.9% | 100.0% | 100.0% |
| **ML: MLP Neural Net (Value-Density)** | 0.5481 | 0.7430 | 94.2% | 100.0% | 100.0% |
| **ML: Learning-to-Rank (LambdaMART)** | 0.5718 | 0.5913 | 75.0% | 100.0% | 100.0% |

## 3. Multi-Model Classification Metrics (Held-Out Test Set)

| Model Name | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.2034 | 0.1182 | 0.1495 | 0.1485 | 0.5590 | 0.1340 |
| **Random Forest** | 1.0000 | 0.0049 | 0.0098 | 0.1019 | 0.4554 | 0.1087 |
| **XGBoost** | 0.1245 | 0.1576 | 0.1391 | 0.1113 | 0.4844 | 0.1665 |
| **Multilayer Perceptron (MLP)** | 0.0000 | 0.0000 | 0.0000 | 0.1206 | 0.5281 | 0.0985 |
| **Learning-to-Rank (LambdaMART)** | 0.1303 | 0.2709 | 0.1760 | 0.1219 | 0.5462 | 0.2040 |