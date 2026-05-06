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

## Setup

```bash
pip install -r requirements.txt
```
