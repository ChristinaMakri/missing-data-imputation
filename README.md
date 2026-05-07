# Missing Data Imputation — A Comparative Study

A research-oriented project comparing missing data imputation methods across different missing mechanisms and data types.

## Research Questions

1. How does each imputation method perform under different missing mechanisms (MCAR, MAR, MNAR)?
2. Does the data type (numerical vs. mixed) affect relative method performance?
3. How does imputation quality impact downstream ML model performance?

## Datasets

| Dataset | Type | Source |
|---|---|---|
| California Housing | Numerical | `sklearn.datasets` |
| Adult Census Income | Mixed (numerical + categorical) | UCI ML Repository |

## Missing Mechanisms

- **MCAR** — Missing Completely At Random
- **MAR** — Missing At Random (depends on observed variables)
- **MNAR** — Missing Not At Random (depends on the missing value itself)

## Imputation Methods

| Category | Method | Library |
|---|---|---|
| Baseline | Mean / Median / Mode | `sklearn` |
| ML-based | KNN Imputation | `sklearn` |
| ML-based | MissForest | `missingpy` |
| Probabilistic | MICE | `miceforest` |
| Deep Learning | Autoencoder | `pytorch` |

## Evaluation

- **RMSE / MAE** — numerical features
- **Accuracy** — categorical features
- **Distribution comparison** — KS test, visual inspection
- **Downstream performance** — classification/regression after imputation

## Project Structure

```
├── data/
│   ├── raw/            # original datasets
│   └── processed/      # datasets with artificially introduced missingness
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis + missing patterns
│   ├── 02_mechanisms.ipynb   # MCAR / MAR / MNAR generation
│   ├── 03_imputation.ipynb   # applying all methods
│   └── 04_evaluation.ipynb   # comparison and results
├── src/
│   ├── missing_generator.py  # controlled missingness introduction
│   ├── imputers.py           # unified interface for all methods
│   └── evaluation.py         # metrics and visualizations
└── results/
    └── figures/
```

## Results

All experiments use 30% missing rate. Metrics are averaged over the imputed columns.

### Imputation Quality — RMSE (lower is better)

**California Housing (numerical)**

| Method | MCAR | MAR | MNAR |
|---|---|---|---|
| **MissForest** | **3.78** | **0.76** | **0.94** |
| KNN | 5.95 | 1.71 | 2.26 |
| Mean/Median/Mode | 6.18 | 1.84 | 2.54 |
| Autoencoder (mean_fill) | 6.82 | 1.51 | 2.70 |
| MICE | 8.28 | 2.65 | 2.91 |

**Adult Census (mixed — numerical columns only)**

| Method | MCAR | MAR | MNAR |
|---|---|---|---|
| **Mean/Median/Mode** | **2873** | 12.3 | 11965 |
| **MissForest** | 2883 | **12.2** | **11832** |
| KNN | 3052 | 13.2 | 12191 |
| MICE | 3354 | 16.9 | 13562 |
| Autoencoder | 2874–2909 | 73–93 | 11854–11861 |

> The large MCAR/MNAR values for Adult Census are driven by `capital_gain`, which has extreme outliers.

### Downstream Performance

| Dataset & Task | Baseline | Best method | Best score |
|---|---|---|---|
| Housing — Regression RMSE | 0.5806 | MissForest (MNAR) | **0.549** |
| Adult — Classification Accuracy | 0.8567 | MissForest (MAR) | **0.859** |

All methods stay within 1–3% of the no-missing baseline for downstream tasks.

---

## Conclusions

1. **MissForest is the most robust method overall.** It wins on RMSE, KS statistic, and downstream performance in the majority of dataset–mechanism combinations. On MNAR Housing it even *surpasses* the no-missing baseline.

2. **Complexity ≠ better results.** MICE (theoretically strong) underperforms Mean/Median/Mode in several scenarios. Autoencoders fail on mixed-type data because they only operate on numerical columns and miss categorical correlations.

3. **The missing mechanism matters more than the method.** MCAR is the hardest to impute (no structure to exploit), while MAR and MNAR are easier because the pattern carries information that tree-based methods can leverage.

4. **For downstream ML, imputation method choice has limited impact.** All methods lose only 1–3% accuracy/RMSE vs. the complete-data baseline. Simple mean imputation is often sufficient as a preprocessor.

5. **Simple baselines are competitive for MCAR.** When missingness is truly random, Mean/Median/Mode rivals or beats advanced methods (e.g., wins on Adult MCAR RMSE).

6. **Practical recommendation:** Start with Mean/Median/Mode as a baseline. Use MissForest when imputation quality matters. Avoid MICE and Autoencoders without dedicated hyperparameter tuning.

---

## Setup

```bash
pip install -r requirements.txt
python run_experiments.py   # reproduces all results
```
