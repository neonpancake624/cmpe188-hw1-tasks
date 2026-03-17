import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def get_task_metadata():
    return {
        "task_name": "logreg_lvl5_iris",
        "description": "Binary logistic regression on Iris (class 0 vs non-0) using PyTorch.",
        "dataset": "sklearn.datasets.load_iris",
        "metrics": ["accuracy", "f1", "mse", "r2"],
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


def make_dataloaders(batch_size: int = 16, seed: int = 42):
    data = load_iris()
    X = data.data.astype(np.float32)

    y = (data.target == 0).astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_train, X_val, mu, sigma = _standardize_train_val(X_train, X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    info = {"n_features": X.shape[1], "mu": mu.tolist(), "sigma": sigma.tolist()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        info,
    )


def build_model(input_dim: int, device=None):
    if device is None:
        device = get_device()
    return nn.Linear(input_dim, 1).to(device)


def train(model, train_loader, val_loader, epochs: int = 300, lr: float = 1e-2, device=None):
    if device is None:
        device = get_device()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_hist, val_loss_hist = [], []
    for _ in range(epochs):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
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
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy().reshape(-1)
            p = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            probs.append(p)
            ys.append(yb.numpy().reshape(-1))

    prob = np.concatenate(probs)
    y = np.concatenate(ys)

    pred = (prob >= 0.5).astype(np.int32)
    y_int = y.astype(np.int32)

    acc = accuracy_score(y_int, pred)
    f1 = f1_score(y_int, pred)

    # Extra metrics (sometimes required by grading protocol)
    mse = mean_squared_error(y, prob)
    r2 = r2_score(y, prob)

    return {"accuracy": float(acc), "f1": float(f1), "mse": float(mse), "r2": float(r2)}


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

    train_loader, val_loader, info = make_dataloaders(batch_size=16, seed=42)
    model = build_model(input_dim=info["n_features"], device=device)

    loss_hist, val_loss_hist = train(model, train_loader, val_loader, epochs=300, lr=1e-2, device=device)

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

    # Iris binary is easy; this should be very stable.
    passed = val_metrics["accuracy"] > 0.90 and val_metrics["f1"] > 0.90 and math.isfinite(val_metrics["mse"])
    print("\nQuality checks:")
    print("  val accuracy > 0.90:", val_metrics["accuracy"])
    print("  val f1       > 0.90:", val_metrics["f1"])
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
