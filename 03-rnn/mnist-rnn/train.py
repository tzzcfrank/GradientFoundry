import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")

    return open(path, "rb")


def find_file(root: Path, name: str) -> Path:
    candidates = [
        root / name,
        root / f"{name}.gz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find {name} or {name}.gz under {root}"
    )


def load_mnist_images(path: Path) -> np.ndarray:
    with open_maybe_gzip(path) as f:
        header = f.read(16)

        magic, count, rows, cols = struct.unpack(
            ">IIII",
            header,
        )

        if magic != 2051:
            raise ValueError(
                f"Invalid image magic number: {magic}"
            )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8,
        )

    images = data.reshape(
        count,
        rows,
        cols,
    )

    images = images.astype(np.float32) / 255.0

    return images


def load_mnist_labels(path: Path) -> np.ndarray:
    with open_maybe_gzip(path) as f:
        header = f.read(8)

        magic, count = struct.unpack(
            ">II",
            header,
        )

        if magic != 2049:
            raise ValueError(
                f"Invalid label magic number: {magic}"
            )

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8,
        )

    if labels.shape[0] != count:
        raise ValueError(
            f"Expected {count} labels, "
            f"got {labels.shape[0]}"
        )

    return labels.astype(np.int64)


def load_mnist(root: Path):
    train_images_path = find_file(
        root,
        "train-images-idx3-ubyte",
    )

    train_labels_path = find_file(
        root,
        "train-labels-idx1-ubyte",
    )

    test_images_path = find_file(
        root,
        "t10k-images-idx3-ubyte",
    )

    test_labels_path = find_file(
        root,
        "t10k-labels-idx1-ubyte",
    )

    x_train = load_mnist_images(
        train_images_path
    )

    y_train = load_mnist_labels(
        train_labels_path
    )

    x_test = load_mnist_images(
        test_images_path
    )

    y_test = load_mnist_labels(
        test_labels_path
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size=28,
        hidden_size=128,
        num_layers=1,
        num_classes=10,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):
        output, h_n = self.rnn(x)

        last_hidden = h_n[-1]

        logits = self.fc(
            last_hidden
        )

        return logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)

        loss = criterion(
            logits,
            yb,
        )

        batch_size = xb.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            logits.argmax(dim=1) == yb
        ).sum().item()

        total_count += batch_size

    avg_loss = (
        total_loss / total_count
    )

    accuracy = (
        total_correct / total_count
    )

    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            batch_size = xb.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_correct += (
                logits.argmax(dim=1)
                == yb
            ).sum().item()

            total_count += batch_size

        train_loss = (
            total_loss
            / total_count
        )

        train_acc = (
            total_correct
            / total_count
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"test_acc={test_acc:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    default_dataset = (
        Path(__file__).resolve().parents[2]
        / "dataset"
        / "MNIST"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    device = choose_device(
        args.device
    )

    print(
        f"device: {device}"
    )

    print(
        f"dataset: {args.dataset}"
    )

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_mnist(
        args.dataset
    )

    print()
    print("raw dataset:")

    print(
        "  x_train:",
        x_train.shape,
    )

    print(
        "  y_train:",
        y_train.shape,
    )

    print(
        "  x_test :",
        x_test.shape,
    )

    print(
        "  y_test :",
        y_test.shape,
    )

    x_train = torch.from_numpy(
        x_train
    )

    y_train = torch.from_numpy(
        y_train
    )

    x_test = torch.from_numpy(
        x_test
    )

    y_test = torch.from_numpy(
        y_test
    )

    print()
    print("tensor dataset:")

    print(
        "  x_train:",
        x_train.shape,
    )

    print(
        "  y_train:",
        y_train.shape,
    )

    print(
        "  x_test :",
        x_test.shape,
    )

    print(
        "  y_test :",
        y_test.shape,
    )

    train_dataset = TensorDataset(
        x_train,
        y_train,
    )

    test_dataset = TensorDataset(
        x_test,
        y_test,
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
    )

    model = RNNClassifier(
        input_size=28,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=10,
    ).to(device)

    print()
    print(model)

    print()
    print("parameters:")

    total_params = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        total_params += count

        print(
            f"  {name:20s} "
            f"shape={str(tuple(parameter.shape)):18s} "
            f"params={count}"
        )

    print(
        f"total parameters: "
        f"{total_params}"
    )

    print()

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()
