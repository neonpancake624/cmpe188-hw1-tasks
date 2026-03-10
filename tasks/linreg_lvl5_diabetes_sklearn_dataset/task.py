import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_task_metadata():
    return {
        "task_name": "linreg_lvl5_diabetes_sklearn_dataset",
        "description": "Linear regression on sklearn diabetes dataset with manual standardization.",
        "dataset": "sklearn.datasets.load_diabetes",
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


def make_dataloaders(batch_size: int = 64, seed: int = 42):
    data = load_diabetes()
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
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_features": X.shape[1],
    }
    return train_loader, val_loader, info


def build_model(input_dim: int, device=None):
    if device is None:
        device = get_device()
    model = nn.Linear(input_dim, 1)
    return model.to(device)


def train(model, train_loader, val_loader, epochs: int = 300, lr: float = 1e-2, device=None):
    if device is None:
        device = get_device()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    val_loss_history = []

    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()

        loss_history.append(running / max(1, len(train_loader)))

        # validation loss (MSE)
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vrun += criterion(pred, yb).item()
        val_loss_history.append(vrun / max(1, len(val_loader)))

    return loss_history, val_loss_history


def evaluate(model, data_loader, device=None):
    if device is None:
        device = get_device()

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    yhat = np.vstack(preds).reshape(-1)
    y = np.vstack(targets).reshape(-1)

    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat)
    return {"mse": float(mse), "r2": float(r2)}


def predict(model, X: np.ndarray, device=None):
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32)).to(device)
        yhat = model(xt).cpu().numpy()
    return yhat


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

    train_loader, val_loader, info = make_dataloaders(batch_size=64, seed=42)
    model = build_model(input_dim=info["n_features"], device=device)

    loss_hist, val_loss_hist = train(
        model, train_loader, val_loader, epochs=2000, lr=1e-2, device=device
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
        "loss_history": [float(x) for x in loss_hist[-20:]],       # keep short
        "val_loss_history": [float(x) for x in val_loss_hist[-20:]]
    }
    save_artifacts(model, outputs, output_dir="output")

    # Quality gate (diabetes is not super high R2; ~0.35+ is commonly reachable)
    passed = val_metrics["r2"] > 0.35 and math.isfinite(val_metrics["mse"])
    print("\nQuality check: val R2 > 0.35 =", val_metrics["r2"])
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())