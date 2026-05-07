import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score


# ── Imputation quality metrics ──────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def evaluate_numerical(original: pd.Series, imputed: pd.Series, missing_mask: pd.Series) -> dict:
    """Compare imputed values against ground truth only at missing positions."""
    true_vals = original[missing_mask].values
    pred_vals = imputed[missing_mask].values
    ks_stat, ks_pvalue = stats.ks_2samp(original.dropna().values, imputed.values)
    return {
        "rmse": rmse(true_vals, pred_vals),
        "mae": mae(true_vals, pred_vals),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
    }


def evaluate_categorical(original: pd.Series, imputed: pd.Series, missing_mask: pd.Series) -> dict:
    """Compare imputed categories against ground truth only at missing positions."""
    true_vals = original[missing_mask].values
    pred_vals = imputed[missing_mask].values
    return {
        "accuracy": accuracy_score(true_vals, pred_vals),
    }


def evaluate_imputation(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Run metrics for all columns and return a summary DataFrame."""
    records = []
    num_cols = df_original.select_dtypes(include=np.number).columns
    cat_cols = df_original.select_dtypes(exclude=np.number).columns

    for col in num_cols:
        if missing_mask[col].any():
            metrics = evaluate_numerical(df_original[col], df_imputed[col], missing_mask[col])
            records.append({"column": col, "type": "numerical", **metrics})

    for col in cat_cols:
        if missing_mask[col].any():
            metrics = evaluate_categorical(df_original[col], df_imputed[col], missing_mask[col])
            records.append({"column": col, "type": "categorical", **metrics})

    return pd.DataFrame(records)


# ── Downstream evaluation ────────────────────────────────────────────────────

def downstream_regression(df: pd.DataFrame, target_col: str, seed: int = 42) -> float:
    """Train a Random Forest regressor and return mean CV RMSE."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def downstream_classification(df: pd.DataFrame, target_col: str, seed: int = 42) -> float:
    """Train a Random Forest classifier and return mean CV accuracy."""
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean()


# ── Visualizations ───────────────────────────────────────────────────────────

def plot_distribution_comparison(
    original: pd.Series,
    imputed_dict: dict,
    col_name: str,
    save_path: str = None,
):
    """Plot distribution of original vs each imputation method for a column."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(original.dropna(), ax=ax, label="Original", linewidth=2)
    for method_name, imputed in imputed_dict.items():
        sns.kdeplot(imputed, ax=ax, label=method_name, linestyle="--")
    ax.set_title(f"Distribution comparison — {col_name}")
    ax.set_xlabel(col_name)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_metrics_comparison(results: pd.DataFrame, metric: str = "rmse", save_path: str = None):
    """Bar plot comparing a metric across methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    results.pivot(index="column", columns="method", values=metric).plot(kind="bar", ax=ax)
    ax.set_title(f"{metric.upper()} by method and column")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Column")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_missing_heatmap(df: pd.DataFrame, save_path: str = None):
    """Heatmap of missing values in a DataFrame."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing value pattern")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
