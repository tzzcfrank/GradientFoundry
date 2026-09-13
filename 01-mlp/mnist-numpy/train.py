#!/usr/bin/env python3
"""
用纯 NumPy 训练一个最简单的 MLP 来识别 MNIST 手写数字。

目的：
1. 看清楚一次完整训练到底发生了什么：
   forward -> loss -> backward -> update
2. 不使用 PyTorch / TensorFlow 自动求导。
3. 手工写出反向传播，让每个梯度都能对应到公式。

网络结构：

    28x28 图片
        |
      flatten
        |
      784
        |
    Linear(784 -> 128)
        |
      ReLU
        |
    Linear(128 -> 10)
        |
      logits
        |
    Softmax + Cross Entropy

默认数据目录：

    GradientFoundry/dataset/MNIST

默认代码目录：

    GradientFoundry/01-mlp/mnist-numpy/train.py

支持 MNIST 官方 IDX 文件的未压缩和 .gz 两种形式：

    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte

或者：

    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
"""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np


# ============================================================
# 1. 读取 MNIST
# ============================================================

def _find_file(dataset_dir: Path, filename: str) -> Path:
    """
    优先寻找未压缩文件；找不到时再寻找 .gz 文件。
    """
    plain = dataset_dir / filename
    gz = dataset_dir / f"{filename}.gz"

    if plain.exists():
        return plain

    if gz.exists():
        return gz

    raise FileNotFoundError(
        f"找不到 MNIST 文件：\n"
        f"  {plain}\n"
        f"  {gz}\n"
        f"请检查 --dataset-dir 参数和数据文件名。"
    )


def _read_all(path: Path) -> bytes:
    """
    读取普通文件或 gzip 文件。
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()

    return path.read_bytes()


def load_mnist_images(path: Path) -> np.ndarray:
    """
    读取 MNIST image IDX 文件。

    IDX image 文件头：
        magic number: 4 bytes
        image count : 4 bytes
        rows        : 4 bytes
        cols        : 4 bytes

    后面紧跟所有像素，每个像素 uint8，范围 0~255。

    返回：
        shape = (N, 784)
        dtype = float32
        数值范围 = 0~1
    """
    data = _read_all(path)

    # >IIII:
    # > 表示 big-endian
    # I 表示 unsigned int，4 字节
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])

    if magic != 2051:
        raise ValueError(f"{path} 不是合法的 MNIST image 文件，magic={magic}")

    images = np.frombuffer(data, dtype=np.uint8, offset=16)

    expected = count * rows * cols
    if images.size != expected:
        raise ValueError(
            f"{path} 数据长度不正确：expected={expected}, actual={images.size}"
        )

    # MNIST 每张图片 28x28。
    # 为了喂给全连接层，我们直接展平成 784。
    images = images.reshape(count, rows * cols)

    # 归一化到 0~1。
    # float32 比 float64 更符合实际深度学习训练场景，也更省内存。
    images = images.astype(np.float32) / 255.0

    return images


def load_mnist_labels(path: Path) -> np.ndarray:
    """
    读取 MNIST label IDX 文件。

    IDX label 文件头：
        magic number: 4 bytes
        label count : 4 bytes

    后面每个 label 是一个 uint8，取值 0~9。

    返回：
        shape = (N,)
        dtype = int64
    """
    data = _read_all(path)

    magic, count = struct.unpack(">II", data[:8])

    if magic != 2049:
        raise ValueError(f"{path} 不是合法的 MNIST label 文件，magic={magic}")

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)

    if labels.size != count:
        raise ValueError(
            f"{path} 数据长度不正确：expected={count}, actual={labels.size}"
        )

    return labels.astype(np.int64)


def load_mnist(dataset_dir: Path):
    """
    一次性加载训练集和测试集。
    """
    train_images = _find_file(dataset_dir, "train-images-idx3-ubyte")
    train_labels = _find_file(dataset_dir, "train-labels-idx1-ubyte")
    test_images = _find_file(dataset_dir, "t10k-images-idx3-ubyte")
    test_labels = _find_file(dataset_dir, "t10k-labels-idx1-ubyte")

    x_train = load_mnist_images(train_images)
    y_train = load_mnist_labels(train_labels)
    x_test = load_mnist_images(test_images)
    y_test = load_mnist_labels(test_labels)

    return x_train, y_train, x_test, y_test


# ============================================================
# 2. 数学函数
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU(x) = max(0, x)

    小于 0 的值变成 0；
    大于 0 的值保持不变。
    """
    return np.maximum(x, 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    把 logits 转成概率。

    logits shape:
        (batch_size, 10)

    返回的每一行：
        10 个概率
        总和为 1

    为什么先减 max？
        exp(1000) 会数值溢出。
        softmax(x) 和 softmax(x - 常数) 完全等价，
        所以先减掉每行最大值可以增强数值稳定性。
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Cross Entropy Loss。

    对于每个样本，只看正确类别的概率。

    比如正确答案是 7：

        模型给 7 的概率 = 0.90 -> loss 很小
        模型给 7 的概率 = 0.01 -> loss 很大

    单个样本：
        loss = -log(P(correct_class))

    一个 batch：
        对 batch 中所有样本的 loss 取平均。

    这也意味着：
        backward 得到的是“这个 batch 的平均梯度”。
    """
    batch_size = probs.shape[0]

    # 防止 log(0)。
    eps = 1e-12

    correct_probs = probs[np.arange(batch_size), labels]
    losses = -np.log(correct_probs + eps)

    return float(np.mean(losses))


# ============================================================
# 3. MLP
# ============================================================

class MLP:
    """
    一个只有 1 个隐藏层的 MLP：

        X
        |
        |  Linear: X @ W1 + b1
        v
        z1
        |
        |  ReLU
        v
        a1
        |
        |  Linear: a1 @ W2 + b2
        v
        logits
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # W1 shape = (784, 128)
        #
        # 输入：
        #   X shape = (batch, 784)
        #
        # X @ W1:
        #   (batch, 784) @ (784, 128)
        #   = (batch, 128)
        # ----------------------------------------------------
        #
        # 使用 He initialization。
        # 对 ReLU 网络来说，比随便初始化一个很大的随机数稳定得多。
        self.W1 = (
            rng.standard_normal((input_size, hidden_size)).astype(np.float32)
            * np.sqrt(2.0 / input_size)
        )
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)

        # 第二层：
        #
        #   (batch, 128) @ (128, 10)
        #   = (batch, 10)
        self.W2 = (
            rng.standard_normal((hidden_size, output_size)).astype(np.float32)
            * np.sqrt(2.0 / hidden_size)
        )
        self.b2 = np.zeros((1, output_size), dtype=np.float32)

    def forward(self, x: np.ndarray):
        """
        前向传播。

        返回：
            logits
            cache

        cache 保存 backward 需要的中间结果。
        """
        # 第一层 Linear。
        z1 = x @ self.W1 + self.b1

        # 激活函数。
        a1 = relu(z1)

        # 第二层 Linear。
        #
        # 注意：这里没有再做 ReLU。
        # 最后一层输出的是 logits，
        # 后面交给 softmax + cross entropy。
        logits = a1 @ self.W2 + self.b2

        # backward 要用到 x / z1 / a1。
        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
        }

        return logits, cache

    def backward(
        self,
        cache: dict[str, np.ndarray],
        probs: np.ndarray,
        labels: np.ndarray,
    ):
        """
        手工实现反向传播。

        整体方向：

            loss
              ^
            softmax
              ^
            logits
              ^
         W2 / b2 / a1
              ^
            ReLU
              ^
              z1
              ^
         W1 / b1 / x

        我们最终需要的不是 x 的梯度，而是：

            dW1
            db1
            dW2
            db2

        因为这四个才是要学习、要更新的参数。
        """
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]

        batch_size = x.shape[0]

        # ====================================================
        # 第一步：loss 对 logits 的梯度
        # ====================================================
        #
        # Softmax + CrossEntropy 合在一起求导后，
        # 有一个非常漂亮的结果：
        #
        #   dlogits = probs - one_hot(labels)
        #
        # 又因为 batch loss 是平均值，
        # 所以最后除以 batch_size。
        #
        # 举例：
        #
        #   probs = [0.1, 0.7, 0.2]
        #   正确答案 = 1
        #
        #   one_hot = [0, 1, 0]
        #
        #   dlogits =
        #   [0.1, 0.7, 0.2] - [0, 1, 0]
        #   [0.1, -0.3, 0.2]
        #
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), labels] -= 1.0
        dlogits /= batch_size

        # ====================================================
        # 第二步：反传第二个 Linear
        # ====================================================
        #
        # forward:
        #
        #   logits = a1 @ W2 + b2
        #
        # 所以：
        #
        #   dW2 = a1.T @ dlogits
        #   db2 = sum(dlogits)
        #   da1 = dlogits @ W2.T
        #
        self.dW2 = a1.T @ dlogits
        self.db2 = np.sum(dlogits, axis=0, keepdims=True)

        # 梯度继续往前传给 a1。
        da1 = dlogits @ self.W2.T

        # ====================================================
        # 第三步：反传 ReLU
        # ====================================================
        #
        # ReLU:
        #
        #   z > 0  -> 输出 z，导数是 1
        #   z <= 0 -> 输出 0，导数是 0
        #
        # 所以：
        #
        #   dz1 = da1 * (z1 > 0)
        #
        dz1 = da1 * (z1 > 0)

        # ====================================================
        # 第四步：反传第一个 Linear
        # ====================================================
        #
        # forward:
        #
        #   z1 = x @ W1 + b1
        #
        # 所以：
        #
        #   dW1 = x.T @ dz1
        #   db1 = sum(dz1)
        #
        self.dW1 = x.T @ dz1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def step(self, learning_rate: float):
        """
        Gradient Descent / SGD 参数更新：

            W_new = W_old - learning_rate * gradient

        注意：
            backward 只负责“算梯度”。
            step 才真正修改模型参数。
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测类别。

        logits 最大的那个位置，就是模型认为最可能的数字。
        """
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)

    def save(self, path: Path):
        """
        保存模型参数到 .npz。
        """
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )


# ============================================================
# 4. 训练和评估
# ============================================================

def evaluate(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
):
    """
    在测试集上计算平均 loss 和 accuracy。

    评估阶段：
        只 forward
        不 backward
        不更新参数
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = x.shape[0]

    for start in range(0, total_samples, batch_size):
        end = start + batch_size

        xb = x[start:end]
        yb = y[start:end]

        logits, _ = model.forward(xb)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, yb)

        # 当前 batch 的 loss 是平均值。
        # 为了最后得到整个测试集的平均值，
        # 这里先乘回 batch 的样本数。
        total_loss += loss * xb.shape[0]

        pred = np.argmax(logits, axis=1)
        total_correct += np.sum(pred == yb)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    """
    完整训练循环。

    一个 epoch：
        把整个训练集全部看一遍。

    一个 batch：
        1. forward
        2. 算 batch mean loss
        3. backward
        4. 更新一次参数

    所以：
        参数不是每条样本更新一次，
        而是每个 batch 更新一次。
    """
    rng = np.random.default_rng(seed)

    num_train = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------
        # 每个 epoch 开始前把训练样本顺序打乱。
        #
        # 否则每次 batch 都固定拿到相同顺序的数据，
        # 对 SGD 不太理想。
        # ----------------------------------------------------
        indices = rng.permutation(num_train)

        epoch_loss_sum = 0.0
        epoch_correct = 0

        for start in range(0, num_train, batch_size):
            end = start + batch_size

            batch_indices = indices[start:end]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]

            # ================================================
            # 1. Forward
            # ================================================
            logits, cache = model.forward(xb)

            # Softmax 得到每个类别的概率。
            probs = softmax(logits)

            # 当前 batch 的平均 loss。
            loss = cross_entropy_loss(probs, yb)

            # ================================================
            # 2. Backward
            # ================================================
            #
            # backward 会算出：
            #
            #   model.dW1
            #   model.db1
            #   model.dW2
            #   model.db2
            #
            model.backward(cache, probs, yb)

            # ================================================
            # 3. Update
            # ================================================
            #
            # 一个 batch 只更新一次参数。
            #
            model.step(learning_rate)

            # 只是为了打印训练指标。
            batch_count = xb.shape[0]
            epoch_loss_sum += loss * batch_count

            pred = np.argmax(logits, axis=1)
            epoch_correct += np.sum(pred == yb)

        train_loss = epoch_loss_sum / num_train
        train_acc = epoch_correct / num_train

        test_loss, test_acc = evaluate(model, x_test, y_test)

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_acc * 100:6.2f}% | "
            f"test loss={test_loss:.4f} | "
            f"test acc={test_acc * 100:6.2f}%"
        )


# ============================================================
# 5. main
# ============================================================

def parse_args():
    # train.py 位于：
    #
    #   GradientFoundry/01-mlp/mnist-numpy/train.py
    #
    # parents[0] = mnist-numpy
    # parents[1] = 01-mlp
    # parents[2] = GradientFoundry
    project_root = Path(__file__).resolve().parents[2]

    default_dataset_dir = project_root / "dataset" / "MNIST"
    default_model_path = Path(__file__).resolve().parent / "mnist_mlp.npz"

    parser = argparse.ArgumentParser(
        description="Train a simple MNIST MLP with pure NumPy"
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help=f"MNIST 数据目录，默认：{default_dataset_dir}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练多少个 epoch，默认 10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每个 batch 的样本数，默认 64",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="隐藏层神经元数量，默认 128",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="SGD 学习率，默认 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="只使用前 N 条训练数据，适合快速调试；默认使用全部训练集",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=default_model_path,
        help=f"模型保存路径，默认：{default_model_path}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"dataset dir : {args.dataset_dir}")
    print(f"batch size  : {args.batch_size}")
    print(f"hidden size : {args.hidden_size}")
    print(f"learning rate: {args.learning_rate}")
    print()

    x_train, y_train, x_test, y_test = load_mnist(args.dataset_dir)

    if args.max_train_samples is not None:
        n = args.max_train_samples
        x_train = x_train[:n]
        y_train = y_train[:n]

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test : {x_test.shape}")
    print(f"y_test : {y_test.shape}")
    print()

    model = MLP(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10,
        seed=args.seed,
    )

    train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_model)

    print()
    print(f"model saved: {args.save_model}")


if __name__ == "__main__":
    main()

