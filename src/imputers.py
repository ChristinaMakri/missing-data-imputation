import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import miceforest as mf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def impute_mean_median_mode(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Baseline imputation — numerical columns with median, categorical with mode."""
    result = df_test.copy()
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_train.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        num_imputer.fit(df_train[num_cols])
        result[num_cols] = num_imputer.transform(df_test[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(df_train[cat_cols])
        result[cat_cols] = cat_imputer.transform(df_test[cat_cols])

    return result


def impute_knn(df_train: pd.DataFrame, df_test: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """KNN imputation — replaces missing values using the mean of k nearest neighbors."""
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    result = df_test.copy()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(df_train[num_cols])
    result[num_cols] = imputer.transform(df_test[num_cols])

    return result


def impute_missforest(df_train: pd.DataFrame, df_test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """MissForest imputation — uses Random Forest to predict missing values iteratively."""
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    result = df_test.copy()

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=seed),
        random_state=seed,
        max_iter=10,
    )
    imputer.fit(df_train[num_cols])
    result[num_cols] = imputer.transform(df_test[num_cols])

    return result


def impute_mice(df_train: pd.DataFrame, df_test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """MICE imputation — Multiple Imputation by Chained Equations using miceforest."""
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    result = df_test.copy()

    kernel = mf.ImputationKernel(df_train[num_cols], random_state=seed)
    kernel.mice(iterations=5)
    result[num_cols] = kernel.impute_new_data(df_test[num_cols]).complete_data(0)

    return result


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def impute_autoencoder(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    training_strategy: str = "complete_cases",
) -> pd.DataFrame:
    """Autoencoder imputation — learns data distribution and reconstructs missing values.

    training_strategy:
        "complete_cases" — train only on rows with no missing values (cleaner signal)
        "mean_fill"      — fill NaNs with column means before training (more data)
    """
    torch.manual_seed(seed)
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    result = df_test.copy()

    col_means = df_train[num_cols].mean()
    test_filled = df_test[num_cols].fillna(col_means).values.astype(np.float32)

    if training_strategy == "complete_cases":
        train_data = df_train[num_cols].dropna().values.astype(np.float32)
    else:
        train_data = df_train[num_cols].fillna(col_means).values.astype(np.float32)

    train_tensor = torch.tensor(train_data)
    dataset = TensorDataset(train_tensor, train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _Autoencoder(input_dim=len(num_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_filled)
        reconstructed = model(test_tensor).numpy()

    # replace only the positions that were originally NaN
    missing_mask = df_test[num_cols].isna().values
    output = test_filled.copy()
    output[missing_mask] = reconstructed[missing_mask]
    result[num_cols] = output

    return result
