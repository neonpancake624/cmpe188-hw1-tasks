import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def get_task_metadata():
    return {
        "task_name": "linreg_lvl6_california_housing",
        "description": "Linear regression on California Housing using PyTorch, standardization, and SGD.",
        "dataset": "sklearn.datasets.fetch_california_housing",
        "metrics": ["mse", "r2"],
    }


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _standardize_train_val(X_train: np.ndarray, X_val: np.ndarray, eps: float = 1e-8):
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, eps)
    return (X_train - mu) / sigma, (X_val - mu) / sigma, mu, sigma


def make_dataloaders(batch_size: int = 128, seed: int = 42):
    """
    California Housing may download the dataset the first time.
    If it can't download (offline), we exit with a non-zero code.
    """
    try:
        data = fetch_california_housing()
    except Exception as e:
        print("ERROR: Could not fetch California Housing dataset.")
        print("Reason:", repr(e))
        print("If your environment is offline, this dataset may not be available.")
        return None, None, None, 2

    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    X_train, X_val, mu, sigma = _standardize_train_val(X_train, X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    info = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_features": X.shape[1],
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }
    return train_loader, val_loader, info, 0


def build_model(input_dim: int, device=None):
    if device is None:
        device = get_device()
    return nn.Linear(input_dim, 1).to(device)


def train(model, train_loader, val_loader, epochs: int = 800, lr: float = 1e-2, device=None):
    """
    For these tabular linear problems, SGD is stable and reliable.
    """
    if device is None:
        device = get_device()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_hist, val_loss_hist = [], []
    for _ in range(epochs):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run += float(loss.item())
        loss_hist.append(run / max(1, len(train_loader)))

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vrun += float(criterion(model(xb), yb).item())
        val_loss_hist.append(vrun / max(1, len(val_loader)))

    return loss_hist, val_loss_hist


def evaluate(model, data_loader, device=None):
    if device is None:
        device = get_device()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            targets.append(yb.numpy())

    yhat = np.vstack(preds).reshape(-1)
    y = np.vstack(targets).reshape(-1)

    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat)
    return {"mse": float(mse), "r2": float(r2)}


def save_artifacts(model, outputs: dict, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(outputs, f, indent=2)


def main():
    set_seed(42)
    device = get_device()

    print("=" * 60)
    print("Task:", get_task_metadata()["task_name"])
    print("Device:", device)
    print("=" * 60)

    train_loader, val_loader, info, code = make_dataloaders(batch_size=128, seed=42)
    if code != 0:
        return code  # dataset fetch failure

    model = build_model(input_dim=info["n_features"], device=device)

    loss_hist, val_loss_hist = train(
        model, train_loader, val_loader, epochs=800, lr=1e-2, device=device
    )

    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nTrain metrics:", train_metrics)
    print("Val metrics:  ", val_metrics)

    outputs = {
        "metadata": get_task_metadata(),
        "info": info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "loss_tail": [float(x) for x in loss_hist[-10:]],
        "val_loss_tail": [float(x) for x in val_loss_hist[-10:]],
    }
    save_artifacts(model, outputs, output_dir="output")

    # R2 on California Housing with linear regression is often ~0.5–0.6 with standardization.
    passed = val_metrics["r2"] > 0.45 and math.isfinite(val_metrics["mse"])
    print("\nQuality check: val R2 > 0.45 =", val_metrics["r2"])
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())