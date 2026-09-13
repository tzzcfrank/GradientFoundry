#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _find_file(dataset_dir: Path, filename: str) -> Path:
    plain = dataset_dir / filename
    gz = dataset_dir / f"{filename}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"找不到 MNIST 文件: {plain} 或 {gz}")


def _read_all(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def load_mnist_images(path: Path) -> np.ndarray:
    data = _read_all(path)
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"{path} 不是合法的 MNIST image 文件，magic={magic}")

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(path: Path) -> np.ndarray:
    data = _read_all(path)
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path} 不是合法的 MNIST label 文件，magic={magic}")

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels.astype(np.int64)


def load_mnist(dataset_dir: Path):
    x_train = load_mnist_images(_find_file(dataset_dir, "train-images-idx3-ubyte"))
    y_train = load_mnist_labels(_find_file(dataset_dir, "train-labels-idx1-ubyte"))
    x_test = load_mnist_images(_find_file(dataset_dir, "t10k-images-idx3-ubyte"))
    y_test = load_mnist_labels(_find_file(dataset_dir, "t10k-labels-idx1-ubyte"))
    return x_train, y_train, x_test, y_test


class MLP(nn.Module):
    """
    784 -> 128 -> 10

    对照 NumPy 版：
        z1 = x @ W1 + b1
        a1 = relu(z1)
        logits = a1 @ W2 + b2
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        logits = self.fc2(a1)
        return logits


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        batch_count = xb.shape[0]
        total_loss += loss.item() * batch_count

        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == yb).sum().item()
        total_samples += batch_count

    return total_loss / total_samples, total_correct / total_samples


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # PyTorch 默认会累积梯度，因此每个 batch 前先清零。
            optimizer.zero_grad()

            # 1. Forward
            logits = model(xb)

            # 2. Loss
            # CrossEntropyLoss 直接接收 logits，不要自己先 softmax。
            loss = criterion(logits, yb)

            # 3. Backward
            # autograd 自动计算所有可训练参数的梯度。
            loss.backward()

            # 4. Update
            # 使用 SGD 时，本质仍然是 W = W - lr * dW。
            optimizer.step()

            # 以下只是训练指标统计。
            batch_count = xb.shape[0]
            epoch_loss_sum += loss.item() * batch_count

            pred = torch.argmax(logits, dim=1)
            epoch_correct += (pred == yb).sum().item()
            epoch_samples += batch_count

        train_loss = epoch_loss_sum / epoch_samples
        train_acc = epoch_correct / epoch_samples

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_acc * 100:6.2f}% | "
            f"test loss={test_loss:.4f} | "
            f"test acc={test_acc * 100:6.2f}%"
        )


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    default_dataset_dir = project_root / "dataset" / "MNIST"
    default_model_path = Path(__file__).resolve().parent / "mnist_mlp.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--save-model", type=Path, default=default_model_path)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.device)

    x_train_np, y_train_np, x_test_np, y_test_np = load_mnist(args.dataset_dir)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
    )

    model = MLP(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
    )

    print(f"device: {device}")
    print(model)
    print()

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    print(f"model saved: {args.save_model}")


if __name__ == "__main__":
    main()

