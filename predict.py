import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse

# Import from other project files
from model import IDCformer


def predict_future(model_path, data_path, dt_col, target_col, feature_cols, seq_len, pred_len, hidden_dim, device):
    """Loads a trained model and predicts future values."""
    # Load the entire dataset to get the last sequence and the scaler
    df = pd.read_excel(data_path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    # All columns to be used as features (including target)
    all_feature_cols = feature_cols + [target_col]
    data_to_scale = df[all_feature_cols].values

    # Initialize scaler and fit it to the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_to_scale)

    # Load model architecture
    input_dim = len(all_feature_cols)
    model = IDCformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=3,
        output_dim=pred_len,
        seq_len=seq_len
    ).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Get the last available sequence from the data
    last_sequence = data_scaled[-seq_len:]
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        future_preds = model(input_tensor)

    # Create dummy array for inverse transform
    dummy_preds = np.zeros((future_preds.shape[1], scaler.n_features_in_))
    target_col_index = all_feature_cols.index(target_col)
    dummy_preds[:, target_col_index] = future_preds.cpu().numpy().flatten()

    # Rescale the predictions
    future_preds_rescaled = scaler.inverse_transform(dummy_preds)[:, target_col_index]

    # Create a DataFrame for the forecast
    last_time = df[dt_col].iloc[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=15), periods=pred_len, freq='15T')

    predictions_df = pd.DataFrame({
        'Time': future_times,
        'Predicted_Value': future_preds_rescaled
    })

    print(f"\nPredictions for the next {pred_len} time steps:")
    print(predictions_df)

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df['Time'], predictions_df['Predicted_Value'], label='Future Predictions', color='green',
             marker='o')
    plt.title(f'Forecast for the Next {pred_len} Steps')
    plt.xlabel('Time')
    plt.ylabel('Predicted Target Value')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict future values using a trained IDCformer model")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (e.g., idcformer.pth)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the full Excel data file')
    parser.add_argument('--dt_col', type=str, default='datetime', help='Name of the datetime column')
    parser.add_argument('--target_col', type=str, default='target', help='Name of the target column to predict')
    parser.add_argument('--feature_cols', nargs='+', default=['feature_1'], help='List of other feature columns')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length used during training')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon used during training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Model hidden dimension')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_future(
        model_path=args.model_path,
        data_path=args.data_path,
        dt_col=args.dt_col,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        device=device
    )