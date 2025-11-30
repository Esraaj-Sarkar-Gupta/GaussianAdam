#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mnist_benchmark.py

MNIST MLP benchmark for FM216 optimizer project.

We keep the same optimizer interface as the other benchmarks:
    test_mnist(opt)  where  opt(x0, grad_fn, f_fn, max_iters, tol) -> dict

Uses:
- 1-hidden-layer ReLU MLP
- Cross-entropy loss
- Mini-batch SGD-style training driven by your optimizer.
"""

import numpy as np

# ---- MNIST LOADER --------------------------------------------------------- #

def load_mnist(flatten: bool = True, limit_train: int | None = None, limit_test: int | None = None):
    """
    Load MNIST data.

    This implementation uses tensorflow.keras.datasets.mnist for convenience.
    If you don't have TensorFlow installed, either:
      - pip install tensorflow
      - or replace this loader with your own.

    Returns:
        X_train, y_train, X_test, y_test
        X_* are float32 in [0, 1], flattened to (N, 784) if flatten=True.
    """
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to load MNIST in this helper.\n"
            "Either install tensorflow or replace load_mnist() with your own loader."
        ) from e

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalize to [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test  = X_test.reshape(-1, 28 * 28)

    if limit_train is not None:
        X_train = X_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test is not None:
        X_test = X_test[:limit_test]
        y_test = y_test[:limit_test]

    return X_train, y_train, X_test, y_test

# ---- MLP PARAM PACK / UNPACK ---------------------------------------------- #

def init_mlp_params(input_dim: int, hidden_dim: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize a 1-hidden-layer MLP:
        input_dim -> hidden_dim (ReLU) -> num_classes (linear)
    """
    # He-style init for ReLU-ish stability
    W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes))
    b2 = np.zeros(num_classes, dtype=np.float32)

    return pack_params(W1, b1, W2, b2)


def pack_params(W1, b1, W2, b2) -> np.ndarray:
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def unpack_params(theta: np.ndarray, input_dim: int, hidden_dim: int, num_classes: int):
    """
    Inverse of pack_params.
    """
    idx = 0
    W1_size = input_dim * hidden_dim
    W1 = theta[idx: idx + W1_size].reshape(input_dim, hidden_dim)
    idx += W1_size

    b1 = theta[idx: idx + hidden_dim]
    idx += hidden_dim

    W2_size = hidden_dim * num_classes
    W2 = theta[idx: idx + W2_size].reshape(hidden_dim, num_classes)
    idx += W2_size

    b2 = theta[idx: idx + num_classes]

    return W1, b1, W2, b2

# ---- MLP FORWARD / LOSS / GRAD ------------------------------------------- #

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def loss_and_grad(theta: np.ndarray,
                  X_batch: np.ndarray,
                  y_batch: np.ndarray,
                  input_dim: int,
                  hidden_dim: int,
                  num_classes: int):
    """
    Compute cross-entropy loss and gradient for the MLP on one mini-batch.
    """
    W1, b1, W2, b2 = unpack_params(theta, input_dim, hidden_dim, num_classes)

    # Forward
    z1 = X_batch @ W1 + b1  # (B, hidden_dim)
    h1 = np.maximum(z1, 0.0)  # ReLU
    logits = h1 @ W2 + b2  # (B, num_classes)
    probs = softmax(logits)

    B = X_batch.shape[0]
    # cross-entropy
    eps = 1e-12
    log_likelihood = -np.log(probs[np.arange(B), y_batch] + eps)
    loss = log_likelihood.mean()

    # Backward
    dlogits = probs
    dlogits[np.arange(B), y_batch] -= 1.0
    dlogits /= B

    dW2 = h1.T @ dlogits
    db2 = dlogits.sum(axis=0)

    dh1 = dlogits @ W2.T
    dz1 = dh1
    dz1[z1 <= 0.0] = 0.0

    dW1 = X_batch.T @ dz1
    db1 = dz1.sum(axis=0)

    grad = pack_params(dW1, db1, dW2, db2)
    return loss, grad

# ---- MNIST BENCHMARK ------------------------------------------------------ #

def test_mnist(
    optimizer,
    max_iters: int = 2000,
    batch_size: int = 128,
    hidden_dim: int = 128,
    seed: int = 0,
    limit_train = None,
    limit_test = None,
):
    """
    MNIST benchmark using your optimizer interface.

    optimizer: callable
        Signature: opt(x0, grad_fn, f_fn, max_iters, tol) -> result_dict
        This is what make_opt(name, **kwargs) returns in main.py.

    Returns:
        dict with keys:
            - final_f    : final (mini-batch) loss
            - n_iters    : number of optimizer steps
            - trace      : list/array of loss values over iterations
            - test_acc   : accuracy on held-out test set
            - noise_log  : GaussianAdam noise log if available
    """
    rng = np.random.default_rng(seed)

    X_train, y_train, X_test, y_test = load_mnist(
        flatten=True,
        limit_train=limit_train,
        limit_test=limit_test,
    )

    n_train, input_dim = X_train.shape
    num_classes = 10

    # Initialize parameters
    theta0 = init_mlp_params(input_dim, hidden_dim, num_classes, rng)

    # Batch sampler
    def sample_batch():
        idx = rng.integers(0, n_train, size=batch_size)
        return X_train[idx], y_train[idx]

    # Cache so that f_fn and grad_fn share the same mini-batch per step
    cache = {"grad": None}

    trace = []

    def f_wrapped(theta):
        Xb, yb = sample_batch()
        loss, grad = loss_and_grad(theta, Xb, yb, input_dim, hidden_dim, num_classes)
        cache["grad"] = grad
        trace.append(loss)
        return loss

    def g_wrapped(theta):
        if cache["grad"] is None:
            # Fallback: compute fresh grad if f_fn wasn't called
            Xb, yb = sample_batch()
            _, grad = loss_and_grad(theta, Xb, yb, input_dim, hidden_dim, num_classes)
        else:
            grad = cache["grad"]
        cache["grad"] = None
        return grad

    # Run optimization via your generic optimizer wrapper
    res = optimizer(theta0, g_wrapped, f_wrapped, max_iters=max_iters, tol=0.0)
    theta_final = res["x"]

    # Compute test accuracy using final parameters
    W1, b1, W2, b2 = unpack_params(theta_final, input_dim, hidden_dim, num_classes)
    z1 = X_test @ W1 + b1
    h1 = np.maximum(z1, 0.0)
    logits = h1 @ W2 + b2
    probs = softmax(logits)
    y_pred = np.argmax(probs, axis=1)
    test_acc = (y_pred == y_test).mean()

    out = {
        "final_loss": trace[-1] if len(trace) > 0 else None,
        "n_iters": len(trace),
        "trace": np.array(trace),
        "test_acc": float(test_acc),
        "noise_log": res.get("noise_log", None),
    }
    
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    # expect (60000, 784) and (10000, 784)
    
    return out

# This file was generated using GPT5.1