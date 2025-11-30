#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helpers.py

Helper utilities: matplotlib plotting and pretty printing
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- Shared colour map so everything is consistent ---- #
COLOR_MAP = {
    "Adam":         "C0",
    "GaussianAdam": "red",
    "AdaGrad":      "C1",
    "RMSProp":      "C2",
    "NAG":          "C4",
    "SGD":          "C5",
    "myOpti":       "C6",
}


def plot_trace(trace, label, ax=None, logy=True):
    if ax is None:
        fig, ax = plt.subplots()
    color = COLOR_MAP.get(label, None)
    ax.plot(trace, label=label, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.1)
    ax.legend()
    return ax


def plot_comparison(traces_dict : dict, title : str, logy=True, logx=False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, tr in traces_dict.items():
        if tr is not None:
            color = COLOR_MAP.get(name, None)
            ax.plot(tr, label=name, color=color)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def print_result(name : str, result) -> None:
    final_f = result.get("final_f", None)
    n_iters = result.get("n_iters", None)

    # format the final objective safely
    if final_f is None or not np.isfinite(final_f):
        f_str = "NaN/Inf"
    else:
        f_str = f"{final_f:.4e}"

    print(f"{name} -> f={f_str}, iters={n_iters}")

def half_gaussian_expected_value(sigma : float, dim : int):
    return sigma * np.sqrt(2 / np.pi) * np.sqrt(dim)

def plot_noise(noise_log : np.ndarray,
               title="GaussianAdam noise (L2 norm per iteration)",
               gamma : float = None,
               expectation_line = False
               ) -> None:
    """
    Plot how the magnitude of GaussianAdam's noise term evolves over iterations.

    noise_log: list of 1D numpy arrays (one per step), as stored in myGaussianAdam.noise_log
    """
    if noise_log is None or len(noise_log) == 0:
        print("No noise_log found (did you run GaussianAdam?).")
        return None, None

    # e.g. plot L2 norm of noise each step
    norms = [np.linalg.norm(z) for z in noise_log]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(norms, color='red', label = f"Gamma = {gamma}")
    
    if (expectation_line == True) and (gamma is not None):
        number_of_iterations = len(noise_log)
        dim = noise_log[0].shape[0]
        sigma_0 = 1 # Okay to make this assumption without double checking
        expected_noise_values = list([])
        
        sigma = sigma_0
        for _ in range(number_of_iterations):
            expected_noise_values.append(half_gaussian_expected_value(sigma, dim))
            sigma *= (1 - gamma)
        
        ax.plot(range(number_of_iterations), expected_noise_values, color = 'orange', label = "Expected value", linestyle="--")
        
    
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖noise‖₂")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_time_bars(time_dict : dict, title : str, logy=False) -> None:
    """
    time_dict: {optimizer_name: elapsed_time_in_seconds}
    
    Plots a bar chart of runtimes, reusing COLOR_MAP for consistency.
    """
    if not time_dict:
        print("No timing data to plot.")
        return None, None

    names = list(time_dict.keys())
    times = [time_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLOR_MAP.get(n, None) for n in names]

    ax.bar(names, times, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Time (s)")

    if logy:
        ax.set_yscale("log")

    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.3g}", ha="center", va="bottom", fontsize=8)

    ax.grid(True, axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax


 # ---- Box Plot Functions ----#
def pretty_print_stats(stats: dict, title: str = "Optimizer Summary"):
    # Title block
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    # Table header
    header = (
        f"{'Benchmark':<12} | "
        f"{'Mean Iters':>12} | "
        f"{'Std Iters':>10} | "
        f"{'Mean Time (s)':>14} | "
        f"{'Std Time (s)':>14}"
    )
    print(header)
    print("-" * 70)

    # Table rows
    for name, s in stats.items():
        row = (
            f"{name:<12} | "
            f"{s['mean_iters']:>12.2f} | "
            f"{s['std_iters']:>10.2f} | "
            f"{s['mean_time']:>14.5f} | "
            f"{s['std_time']:>14.5f}"
        )
        print(row)

    print("=" * 70 + "\n")

 
def plot_boxplots(
        raw_iters: dict,
        raw_times: dict,
        title_prefix: str = "Optimizer Results",
        figsize: tuple = (8, 5)
    ):
    benchmarks = list(raw_iters.keys())

    # --- ITERATIONS BOXPLOT --- #
    plt.figure(figsize=figsize)
    data_iters = [raw_iters[b] for b in benchmarks]
    plt.boxplot(data_iters, labels=benchmarks, showmeans=True)
    plt.title(f"{title_prefix} — Iterations to Converge")
    plt.xlabel("Benchmark")
    plt.ylabel("Iterations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- RUNTIME BOXPLOT --- #
    plt.figure(figsize=figsize)
    data_times = [raw_times[b] for b in benchmarks]
    plt.boxplot(data_times, labels=benchmarks, showmeans=True)
    plt.title(f"{title_prefix} — Runtime Distribution")
    plt.xlabel("Benchmark")
    plt.ylabel("Time (seconds)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# This file was generated using GPT5.1