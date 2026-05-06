import numpy as np
import pandas as pd


def introduce_mcar(df: pd.DataFrame, columns: list, missing_rate: float, seed: int = 42) -> pd.DataFrame:
    """Introduce Missing Completely At Random — missingness is independent of all variables."""
    rng = np.random.default_rng(seed)
    result = df.copy()
    for col in columns:
        mask = rng.random(len(result)) < missing_rate
        result.loc[mask, col] = np.nan
    return result


def introduce_mar(
    df: pd.DataFrame,
    target_col: str,
    condition_col: str,
    missing_rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Introduce Missing At Random — missingness depends on another observed variable."""
    rng = np.random.default_rng(seed)
    result = df.copy()
    threshold = df[condition_col].quantile(0.5)
    high_group = result[condition_col] > threshold
    mask = high_group & (rng.random(len(result)) < missing_rate)
    result.loc[mask, target_col] = np.nan
    return result


def introduce_mnar(
    df: pd.DataFrame,
    target_col: str,
    missing_rate: float,
    direction: str = "high",
    seed: int = 42,
) -> pd.DataFrame:
    """Introduce Missing Not At Random — missingness depends on the value itself."""
    rng = np.random.default_rng(seed)
    result = df.copy()
    col_values = df[target_col].rank(pct=True)
    if direction == "high":
        propensity = col_values * missing_rate * 2
    else:
        propensity = (1 - col_values) * missing_rate * 2
    propensity = propensity.clip(0, 1)
    mask = rng.random(len(result)) < propensity
    result.loc[mask, target_col] = np.nan
    return result
