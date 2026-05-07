"""End-to-end experiment runner — produces numerical results without needing Jupyter."""
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.missing_generator import introduce_mcar, introduce_mar, introduce_mnar
from src.imputers import (
    impute_mean_median_mode,
    impute_knn,
    impute_missforest,
    impute_mice,
    impute_autoencoder,
)
from src.evaluation import evaluate_imputation, downstream_regression, downstream_classification

SEED = 42

# ── 1. Load datasets ──────────────────────────────────────────────────────────
print("Loading data...")
housing = fetch_california_housing(as_frame=True)
df_housing = housing.frame

# Adult Census — try UCI, fall back to sklearn's prepackaged version
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    df_adult = pd.read_csv(url, header=None, names=col_names, na_values=" ?", skipinitialspace=True)
    print(f"Adult Census downloaded from UCI: {df_adult.shape}")
except Exception as e:
    print(f"UCI download failed ({e}), using sklearn fetch_openml...")
    from sklearn.datasets import fetch_openml
    adult_raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df_adult = adult_raw.frame
    df_adult.rename(columns={"class": "income"}, inplace=True)
    print(f"Adult Census from OpenML: {df_adult.shape}")

print(f"Housing: {df_housing.shape} | Adult: {df_adult.shape}")

# ── 2. Save raw data ──────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
df_housing.to_csv("data/raw/housing.csv", index=False)
df_adult.to_csv("data/raw/adult.csv", index=False)

# ── 3. Train / test split ─────────────────────────────────────────────────────
housing_train, housing_test = train_test_split(df_housing, test_size=0.2, random_state=SEED)
adult_train,   adult_test   = train_test_split(df_adult,   test_size=0.2, random_state=SEED)
housing_train = housing_train.reset_index(drop=True)
housing_test  = housing_test.reset_index(drop=True)
adult_train   = adult_train.reset_index(drop=True)
adult_test    = adult_test.reset_index(drop=True)

# ── 4. Methods ────────────────────────────────────────────────────────────────
METHODS = {
    "Mean/Median/Mode": lambda tr, te: impute_mean_median_mode(tr, te),
    "KNN":              lambda tr, te: impute_knn(tr, te),
    "MissForest":       lambda tr, te: impute_missforest(tr, te, seed=SEED),
    "MICE":             lambda tr, te: impute_mice(tr, te, seed=SEED),
    "Autoencoder (complete_cases)": lambda tr, te: impute_autoencoder(tr, te, seed=SEED, training_strategy="complete_cases"),
    "Autoencoder (mean_fill)":      lambda tr, te: impute_autoencoder(tr, te, seed=SEED, training_strategy="mean_fill"),
}

def run_all(train_miss, test_miss):
    results = {}
    for name, fn in METHODS.items():
        print(f"    {name}...")
        results[name] = fn(train_miss, test_miss)
    return results

# ── 5. Introduce missingness & impute ─────────────────────────────────────────
housing_target_cols = ["MedInc", "HouseAge", "AveRooms"]
adult_target_cols   = ["age", "hours_per_week", "capital_gain"]

# Housing
housing_imputed = {}
housing_masks   = {}
for mech in ["MCAR", "MAR", "MNAR"]:
    print(f"\nHousing — {mech}")
    if mech == "MCAR":
        h_miss_train = introduce_mcar(housing_train, columns=housing_target_cols, missing_rate=0.3)
        h_miss_test  = introduce_mcar(housing_test,  columns=housing_target_cols, missing_rate=0.3)
    elif mech == "MAR":
        h_miss_train = introduce_mar(housing_train, target_col="MedInc", condition_col="HouseAge", missing_rate=0.3)
        h_miss_test  = introduce_mar(housing_test,  target_col="MedInc", condition_col="HouseAge", missing_rate=0.3)
    else:
        h_miss_train = introduce_mnar(housing_train, target_col="MedInc", missing_rate=0.3, direction="high")
        h_miss_test  = introduce_mnar(housing_test,  target_col="MedInc", missing_rate=0.3, direction="high")
    housing_masks[mech]   = h_miss_test.isna()
    housing_imputed[mech] = run_all(h_miss_train, h_miss_test)

# Adult
adult_imputed = {}
adult_masks   = {}
for mech in ["MCAR", "MAR", "MNAR"]:
    print(f"\nAdult — {mech}")
    if mech == "MCAR":
        a_miss_train = introduce_mcar(adult_train, columns=adult_target_cols, missing_rate=0.3)
        a_miss_test  = introduce_mcar(adult_test,  columns=adult_target_cols, missing_rate=0.3)
    elif mech == "MAR":
        a_miss_train = introduce_mar(adult_train, target_col="hours_per_week", condition_col="age", missing_rate=0.3)
        a_miss_test  = introduce_mar(adult_test,  target_col="hours_per_week", condition_col="age", missing_rate=0.3)
    else:
        a_miss_train = introduce_mnar(adult_train, target_col="capital_gain", missing_rate=0.3, direction="high")
        a_miss_test  = introduce_mnar(adult_test,  target_col="capital_gain", missing_rate=0.3, direction="high")
    adult_masks[mech]   = a_miss_test.isna()
    adult_imputed[mech] = run_all(a_miss_train, a_miss_test)

# ── 6. Save processed data ────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/housing_imputed.pkl", "wb") as f: pickle.dump(housing_imputed, f)
with open("data/processed/adult_imputed.pkl",   "wb") as f: pickle.dump(adult_imputed,   f)
housing_test.to_csv("data/processed/housing_test_original.csv", index=False)
adult_test.to_csv("data/processed/adult_test_original.csv",     index=False)

# ── 7. Evaluate imputation quality ───────────────────────────────────────────
print("\n\n=== IMPUTATION QUALITY — California Housing ===")
housing_results = []
for mech in ["MCAR", "MAR", "MNAR"]:
    for method_name, df_imp in housing_imputed[mech].items():
        m = evaluate_imputation(housing_test, df_imp, housing_masks[mech])
        m["method"] = method_name
        m["mechanism"] = mech
        housing_results.append(m)
housing_results = pd.concat(housing_results, ignore_index=True)

h_summary = housing_results[housing_results["type"] == "numerical"].groupby(["method", "mechanism"])["rmse"].mean().unstack().round(4)
print(h_summary.to_string())

print("\n\n=== IMPUTATION QUALITY — Adult Census ===")
adult_results = []
for mech in ["MCAR", "MAR", "MNAR"]:
    for method_name, df_imp in adult_imputed[mech].items():
        m = evaluate_imputation(adult_test, df_imp, adult_masks[mech])
        m["method"] = method_name
        m["mechanism"] = mech
        adult_results.append(m)
adult_results = pd.concat(adult_results, ignore_index=True)

a_summary = adult_results[adult_results["type"] == "numerical"].groupby(["method", "mechanism"])["rmse"].mean().unstack().round(4)
print(a_summary.to_string())

# ── 8. KS statistics ──────────────────────────────────────────────────────────
print("\n\n=== KS STATISTIC (lower = distributions closer) — Housing ===")
ks_h = housing_results[housing_results["type"] == "numerical"].groupby(["method", "mechanism"])["ks_stat"].mean().unstack().round(4)
print(ks_h.to_string())

print("\n\n=== KS STATISTIC — Adult ===")
ks_a = adult_results[adult_results["type"] == "numerical"].groupby(["method", "mechanism"])["ks_stat"].mean().unstack().round(4)
print(ks_a.to_string())

# ── 9. Downstream evaluation ──────────────────────────────────────────────────
print("\n\n=== DOWNSTREAM — California Housing (Regression RMSE) ===")
baseline_rmse = downstream_regression(housing_test, target_col="MedHouseVal", seed=SEED)
print(f"Baseline (no missing): {baseline_rmse:.4f}")

downstream_h = []
for mech in ["MCAR", "MAR", "MNAR"]:
    for method_name, df_imp in housing_imputed[mech].items():
        r = downstream_regression(df_imp, target_col="MedHouseVal", seed=SEED)
        downstream_h.append({"method": method_name, "mechanism": mech, "rmse": r})
downstream_h = pd.DataFrame(downstream_h)
print(downstream_h.pivot(index="method", columns="mechanism", values="rmse").round(4).to_string())

print("\n\n=== DOWNSTREAM — Adult Census (Classification Accuracy) ===")
baseline_acc = downstream_classification(adult_test, target_col="income", seed=SEED)
print(f"Baseline (no missing): {baseline_acc:.4f}")

downstream_a = []
for mech in ["MCAR", "MAR", "MNAR"]:
    for method_name, df_imp in adult_imputed[mech].items():
        acc = downstream_classification(df_imp, target_col="income", seed=SEED)
        downstream_a.append({"method": method_name, "mechanism": mech, "accuracy": acc})
downstream_a = pd.DataFrame(downstream_a)
print(downstream_a.pivot(index="method", columns="mechanism", values="accuracy").round(4).to_string())

# ── 10. Save CSVs ─────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
h_summary.to_csv("results/housing_rmse.csv")
a_summary.to_csv("results/adult_rmse.csv")
ks_h.to_csv("results/housing_ks.csv")
ks_a.to_csv("results/adult_ks.csv")
downstream_h.pivot(index="method", columns="mechanism", values="rmse").round(4).to_csv("results/housing_downstream_rmse.csv")
downstream_a.pivot(index="method", columns="mechanism", values="accuracy").round(4).to_csv("results/adult_downstream_accuracy.csv")

print("\n\nResults saved to results/")
