import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_path, dt_col, target_col, feature_cols, sequence_length, forecast_horizon,
                             batch_size=32):
    """
    Loads data from an Excel file, preprocesses it, and creates DataLoaders.
    """
    # Load and sort data
    df = pd.read_excel(file_path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    # All columns to be used as features (including target)
    all_feature_cols = feature_cols + [target_col]
    data_to_scale = df[all_feature_cols].values

    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_to_scale)

    # Find the index of the target column for label creation
    target_col_index = all_feature_cols.index(target_col)

    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - forecast_horizon + 1):
        X.append(data_scaled[i: i + sequence_length])
        y.append(data_scaled[i + sequence_length: i + sequence_length + forecast_horizon, target_col_index])

    X, y = np.array(X), np.array(y)

    # Split data chronologically
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size: train_size + val_size], y[train_size: train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    print(f"Dataset split:")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # Convert to PyTorch tensors
    X_train_tensor, y_train_tensor = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val_tensor, y_val_tensor = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test_tensor, y_test_tensor = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Return the scaler for inverse transform later
    return train_loader, val_loader, test_loader, scaler, target_col_index, df