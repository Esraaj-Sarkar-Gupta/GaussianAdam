#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — clean entry point for FM216 Optimizer Benchmarks.
"""

import matplotlib.pyplot as plt
from time import perf_counter, time
import numpy as np

from pathlib import Path
import os

from benchmarks import (
    test_well_conditioned_quadratic,
    test_ill_conditioned_quadratic,
    test_rosenbrock,
    test_rastrigin, # Non-convex benchmark function
    my_optimizer,
)
from helpers import plot_comparison, print_result, plot_noise, plot_time_bars, plot_boxplots, pretty_print_stats

from mnist_benchmark import test_mnist


# Optimizer wrapper
def make_opt(name, **kwargs):
    return lambda x0, g, f, max_iters=1000, tol=1e-6: my_optimizer(
        x0,
        grad_fn=g,
        f_fn=f,
        max_iters=max_iters,
        algo=name,
        **kwargs,
    )

# Plot the Gaussian Noise
def inline_plot_noise(name, res, opti_name):
    if name == "GaussianAdam" and res.get("noise_log") is not None:
        gamma = res.get("gamma", None)
        gamma = 0.001
        plot_noise(
            res["noise_log"],
            title=f"GaussianAdam noise -- {opti_name}",
            gamma = gamma,
            expectation_line = True
            )
    

# Main function
def run_all(test_beds : list[str]) -> None:

    # List of optimizers
    optimizers = {
        "AdaGrad":     make_opt("adagrad"),
        #"RMSProp":     make_opt("rmsprop"),
        "Adam":         make_opt("adam"),
        "GaussianAdam": make_opt("mygaussianadam"),
        #"NAG":        make_opt("nag"),
        #"SGD":        make_opt("sgd"),
    }
    
    if "wcQ" in test_beds:
        # ---- Well-Conditioned Quadratic ---- #
        traces = {}
        times = {}   # timing per optimizer
        print("=== Well-Conditioned Quadratic ===")
    
        for name, opt in optimizers.items():
            t0 = perf_counter()
            res = test_well_conditioned_quadratic(opt)
            elapsed = perf_counter() - t0
    
            print_result(name, res)
            traces[name] = res["trace"]
            times[name] = elapsed
    
            inline_plot_noise(name, res, "Well-Conditioned Quadratic")
    
        plot_comparison(traces, "Well-Conditioned Quadratic")
        plt.show()
    
        plot_time_bars(times, "Runtime — Well-Conditioned Quadratic")
        plt.show()
    
    if "icQ" in test_beds:
        # ---- Ill-Conditioned Quadratic ---- #
        traces = {}
        times = {}
        
        print("=== Ill-Conditioned Quadratic ===")
    
        for name, opt in optimizers.items():
            t0 = perf_counter()
            res = test_ill_conditioned_quadratic(opt)
            elapsed = perf_counter() - t0
    
            print_result(name, res)
            traces[name] = res["trace"]
            times[name] = elapsed
    
            inline_plot_noise(name, res, "Ill-Conditioned Quadratic")
    
        plot_comparison(traces, "Ill-Conditioned Quadratic")
        plt.show()
    
        plot_time_bars(times, "Runtime — Ill-Conditioned Quadratic")
        plt.show()
    
    if "RsB" in test_beds:
    
        # ---- Rosenbrock Function ---- #
        traces = {}
        times = {}
        print("=== Rosenbrock ===")
    
        for name, opt in optimizers.items():
            t0 = perf_counter()
            res = test_rosenbrock(opt)
            elapsed = perf_counter() - t0
    
            print_result(name, res)
            traces[name] = res["trace"]
            times[name] = elapsed
    
            inline_plot_noise(name,res, "Rosenbrock Function")
    
        plot_comparison(traces, "Rosenbrock Function")
        plt.show()
    
        plot_time_bars(times, "Runtime -- Rosenbrock Function")
        plt.show()
        
    if "Rast" in test_beds:
        # ---- Rastrigin Function ---- #
        traces = {}
        times = {}
        print("=== Rastrigin ===")
        
        for name, opt in optimizers.items():
            t0 = perf_counter()
            res = test_rastrigin(opt)
            elapsed = perf_counter() - t0
        
            print_result(name, res)
            traces[name] = res["trace"]
            times[name] = elapsed

        inline_plot_noise(name, res, "Rastrigin")

        plot_comparison(traces, "Rastrigin Function")
        plt.show()

        plot_time_bars(times, "Runtime — Rastrigin Function")
        plt.close()
    
    if "MNIST" in test_beds:
        # ---- MNIST MLP ---- #
        traces = {}
        times = {}
        print("=== MNIST MLP ===")
    
        for name, opt in optimizers.items():
            t0 = perf_counter()
            res = test_mnist(opt, max_iters=5000)   # uses default max_iters, etc.
            elapsed = perf_counter() - t0
    
            print_result(name, res)  # prints final_f and n_iters
    
            test_acc = res.get("test_acc", None)
            if test_acc is not None:
                print(f"{name} -> test_acc={test_acc:.4f}")
    
            traces[name] = res["trace"]
            times[name] = elapsed
            
        inline_plot_noise(name, res, "MNIST")
    
        # Loss trace (no log scale; cross-entropy is ~O(1))
        plot_comparison(traces, "MNIST MLP — Training Loss", logy=True, logx=True)
        plt.show()
    
        # Runtime comparison
        plot_time_bars(times, "Runtime — MNIST MLP")
        plt.show()

    
    
def test_rastrigin_params(
        alpha_list : np.ndarray,
        std_list : np.ndarray,
        gamma_list : np.ndarray
        ) -> np.ndarray :
    print("Running rastigin function paramater search with GaussianAdam")
    times = np.array([])
    final_f_values = np.array([])
    
    file = Path("params.csv")
    try:
        os.remove(file)
    except:
        pass
    f = file.open("a")
    
    f.write("alpha,std,gamma,final_f\n")
    
    for alpha in alpha_list:
        for std in std_list:
            for gamma in gamma_list:
                start_time = perf_counter()
                res = test_rastrigin(make_opt("mygaussianadam", alpha = alpha, gaussian_std = std, gamma = gamma))
                delta_time = perf_counter() - start_time
                
                final_f = res["final_f"]
                
                np.append(times, delta_time)
                np.append(final_f_values, final_f)
                
                f.write(f"{alpha},{std},{gamma},{final_f}\n")
    f.close()  
    print(f"Process finished with {len(times)} iterations, with an average of {np.mean(times)}s per iteration.")
    
    plt.plot(len(final_f_values), final_f_values, color = 'orange')

def run_multiple(
        optimizer_factory,
        benchmark_names: list[str],
        n_runs: int = 10,
        max_iters: int = 1000,
        tol: float = 1e-6
    ):

    # Map benchmark names to their functions
    bench_map = {
        "wcQ": test_well_conditioned_quadratic,
        "icQ": test_ill_conditioned_quadratic,
        "RsB": test_rosenbrock,
        "Rast": test_rastrigin,
        "MNIST": test_mnist,
    }

    results = {}

    for name in benchmark_names:
        if name not in bench_map:
            raise ValueError(f"Unknown benchmark '{name}'")

        test_fn = bench_map[name]

        iters_list = []
        time_list = []

        print(f"\n=== Running {name} for {n_runs} runs ===")

        for _ in range(n_runs):
            t0 = perf_counter()
            res = test_fn(optimizer_factory, max_iters=max_iters)
            elapsed = perf_counter() - t0

            # Store
            iters_list.append(res["n_iters"])
            time_list.append(elapsed)

        iters_arr = np.array(iters_list, dtype=float)
        time_arr = np.array(time_list, dtype=float)

        results[name] = {
            "mean_iters": float(iters_arr.mean()),
            "std_iters": float(iters_arr.std()),
            "mean_time": float(time_arr.mean()),
            "std_time": float(time_arr.std()),
        }

    return results

"""
def run_multiple(
        optimizer_factory,
        benchmark_names: list[str],
        n_runs: int = 10,
        max_iters: int = 1000,
        tol: float = 1e-6
    ):
    # Map benchmark names to their functions
    bench_map = {
        "wcQ": test_well_conditioned_quadratic,
        "icQ": test_ill_conditioned_quadratic,
        "RsB": test_rosenbrock,
        "Rast": test_rastrigin,
        "MNIST": test_mnist,
    }

    results = {}

    for name in benchmark_names:
        if name not in bench_map:
            raise ValueError(f"Unknown benchmark '{name}'")

        test_fn = bench_map[name]

        iters_list = []
        time_list = []

        print(f"\n=== Running {name} for {n_runs} runs ===")

        for _ in range(n_runs):
            t0 = perf_counter()
            res = test_fn(optimizer_factory, max_iters=max_iters)
            elapsed = perf_counter() - t0

            # Store
            iters_list.append(res["n_iters"])
            time_list.append(elapsed)

        iters_arr = np.array(iters_list, dtype=float)
        time_arr = np.array(time_list, dtype=float)

        results[name] = {
            "mean_iters": float(iters_arr.mean()),
            "std_iters": float(iters_arr.std()),
            "mean_time": float(time_arr.mean()),
            "std_time": float(time_arr.std()),
        }

    return results
"""

import numpy as np
from time import perf_counter

def run_multiple(
    optimizer_factory,
    benchmark_names: list[str],
    n_runs: int = 10,
    max_iters: int = 1000,
    tol: float = 1e-6
    ):
    """
    Run optimizers and collect specific stats based on the benchmark type.
    """

    # Map benchmark names to their functions
    # Ensure your test_mnist function returns a dict like: 
    # {'final_loss': float, 'test_acc': float}
    bench_map = {
        "wcQ": test_well_conditioned_quadratic,
        "icQ": test_ill_conditioned_quadratic,
        "RsB": test_rosenbrock,
        "Rast": test_rastrigin,
        "MNIST": test_mnist,
    }

    results = {}

    for name in benchmark_names:
        if name not in bench_map:
            raise ValueError(f"Unknown benchmark '{name}'")

        test_fn = bench_map[name]
        
        # Initialize lists for standard Benchmarks
        iters_list = []
        time_list = []
        
        # Initialize lists for MNIST
        loss_list = []
        acc_list = []

        print(f"\n=== Running {name} for {n_runs} runs ===")

        for i in range(n_runs):
            t0 = perf_counter()
            # Pass strict args only if your test functions accept them
            res = test_fn(optimizer_factory, max_iters=max_iters)
            elapsed = perf_counter() - t0

            # --- CONDITIONAL LOGIC ---
            if name == "MNIST":
                # For MNIST, we care about Loss and Accuracy
                # Assuming res looks like: {'final_loss': 0.2, 'test_acc': 0.95}
                loss_list.append(res.get("final_loss", 0.0))
                acc_list.append(res.get("test_acc", 0.0))
                print(f"  Run {i+1}: Acc={res.get('test_acc'):.4f}, Loss={res.get('final_loss'):.4f}")
            else:
                # For Standard Functions, we care about Speed/Iters
                iters_list.append(res.get("n_iters", max_iters))
                time_list.append(elapsed)

        # --- COMPUTE STATISTICS ---
        if name == "MNIST":
            loss_arr = np.array(loss_list, dtype=float)
            acc_arr = np.array(acc_list, dtype=float)
            
            results[name] = {
                "type": "ML_Metric", # Tag to identify later
                "mean_loss": float(loss_arr.mean()),
                "std_loss": float(loss_arr.std()),
                "mean_acc": float(acc_arr.mean()),
                "std_acc": float(acc_arr.std()),
                # Save raw data for plotting later if needed
                "raw_loss": loss_list,
                "raw_acc": acc_list
            }
        else:
            iters_arr = np.array(iters_list, dtype=float)
            time_arr = np.array(time_list, dtype=float)
            
            results[name] = {
                "type": "Convergence_Metric",
                "mean_iters": float(iters_arr.mean()),
                "std_iters": float(iters_arr.std()),
                "mean_time": float(time_arr.mean()),
                "std_time": float(time_arr.std()),
                "raw_iters": iters_list,
                "raw_time": time_list
            }

    return results      
            

# Main :D
if __name__ == "__main__":
    # ---- RUn Optimizers on Benchmark ---- #
    run_all(["wcQ, icQ, RsB, Rast"])
    
    # ---- Rastrigin Hyperparameter Space Search ---- #
    #alpha_list = np.linspace(0.01, 5, 100)
    #std_list = np.arange(1, 10+1, 1)
    #gamma_list = np.linspace(0.001, 0.01, 100)
    
    #test_rastrigin_params(alpha_list, std_list, gamma_list)
    
    # ---- Gaussian Adam -- Statistics ---- #
    """
    N_datapoints = 10
    
    gaussian_opt = make_opt(
        "mygaussianadam"
    )

    benches = ["Rast"]

    # Compute summary stats
    stats = run_multiple(
        optimizer_factory=gaussian_opt,
        benchmark_names=benches,
        n_runs= N_datapoints, 
        max_iters=3500
    )
    print(stats)

    # Collect raw values for boxplots
    raw_iters = {name: [] for name in benches}
    raw_times = {name: [] for name in benches}

    for name in benches:
        test_fn = {
            "wcQ": test_well_conditioned_quadratic,
            "icQ": test_ill_conditioned_quadratic,
            "RsB": test_rosenbrock,
            "Rast": test_rastrigin,
            "MNIST": test_mnist
        }[name]

        for _ in range(N_datapoints):
            t0 = perf_counter()
            res = test_fn(gaussian_opt, max_iters=5000)
            elapsed = perf_counter() - t0

            raw_iters[name].append(res["n_iters"])
            raw_times[name].append(elapsed)
            
    # Pretty print
    pretty_print_stats(stats, title=f"GaussianAdam — Summary ({N_datapoints} runs)")

    # Plot
    plot_boxplots(
        raw_iters,
        raw_times,
        title_prefix=f"GaussianAdam ({N_datapoints} runs)"
    )
    """
    # ---- GaussianAdam MNIST -- Statistical Analysis ---- #
    """
    N_datapoints = 5 # Reduced to 5 for MNIST (it's slow!)
    
    gaussian_opt = make_opt("mygaussianadam")
    benches = ["MNIST"] # Or ["Rast", "MNIST"]

    # 1. Run Benchmark ONCE
    stats = run_multiple(
        optimizer_factory=gaussian_opt,
        benchmark_names=benches,
        n_runs=N_datapoints, 
        max_iters=2000 # Adjust as needed for MNIST
    )

    # 2. Print Summary
    print("-" * 30)
    print("FINAL RESULTS")
    print("-" * 30)
    
    for name, data in stats.items():
        print(f"Benchmark: {name}")
        if data["type"] == "ML_Metric":
            print(f"  Mean Loss: {data['mean_loss']:.4f} ± {data['std_loss']:.4f}")
            print(f"  Mean Acc:  {data['mean_acc']:.2%} ± {data['std_acc']:.2%}")
        else:
            print(f"  Mean Iters: {data['mean_iters']:.2f} ± {data['std_iters']:.2f}")
            print(f"  Mean Time:  {data['mean_time']:.4f}s")
    """

