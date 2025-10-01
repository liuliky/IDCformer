import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import argparse
import os

# Import from other project files
from model import IDCformer
from data_loader import load_and_preprocess_data


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Handles the model training loop."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting model training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs, val_y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

    print('Training complete.')
    return model


def evaluate_model(model, test_loader, scaler, target_col_index, device):
    """Evaluates the model on the test set and prints metrics."""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X = test_X.to(device)
            preds = model(test_X)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(test_y.numpy())

    y_pred_np = np.concatenate(all_preds)
    y_true_np = np.concatenate(all_trues)

    # Create dummy arrays for inverse transform
    dummy_pred = np.zeros((y_pred_np.shape[0] * y_pred_np.shape[1], scaler.n_features_in_))
    dummy_true = np.zeros((y_true_np.shape[0] * y_true_np.shape[1], scaler.n_features_in_))

    dummy_pred[:, target_col_index] = y_pred_np.flatten()
    dummy_true[:, target_col_index] = y_true_np.flatten()

    # Rescale to original values
    y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_col_index]
    y_true_rescaled = scaler.inverse_transform(dummy_true)[:, target_col_index]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)

    print(f"\nTest Set Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(y_true_rescaled, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(y_pred_rescaled, label='Predicted Values', color='red', linestyle='--')
    plt.title('Test Set: Actual vs. Predicted Target Value')
    plt.xlabel('Time Steps')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="IDCformer for Time Series Forecasting")
    parser.add_argument('--data_path', type=str, default='data/your_data.xlsx', help='Path to your Excel data file')
    parser.add_argument('--dt_col', type=str, default='datetime', help='Name of the datetime column')
    parser.add_argument('--target_col', type=str, default='target', help='Name of the target column to predict')
    parser.add_argument('--feature_cols', nargs='+', default=['feature_1'], help='List of other feature columns')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_save_path', type=str, default='idcformer.pth', help='Path to save the trained model')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load data
    train_loader, val_loader, test_loader, scaler, target_col_index, _ = load_and_preprocess_data(
        file_path=args.data_path,
        dt_col=args.dt_col,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        sequence_length=args.seq_len,
        forecast_horizon=args.pred_len,
        batch_size=args.batch_size
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = len(args.feature_cols) + 1
    model = IDCformer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_layers=3,
        output_dim=args.pred_len,
        seq_len=args.seq_len
    ).to(device)

    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )

    # Save the trained model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), args.model_save_path)
    print(f"Trained model saved to {args.model_save_path}")

    # Evaluate model
    evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        scaler=scaler,
        target_col_index=target_col_index,
        device=device
    )


if __name__ == '__main__':
    main()