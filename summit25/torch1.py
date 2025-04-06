import torch
import time
import os
import gc
import matplotlib.pyplot as plt
import argparse
import sys

def matrix_vector_operations(N_values):

    if args.verbose:
        print(f"Running test on : {device}  with type : {dtype} matrix_mode : {args.matrix_mode}  number of runs : {runs}")

    results = {}
    overall_time = {}

    for N in N_values:

      if matrix_mode:
          second_dimension = N
      else:
          second_dimension = 1

      results[N] = 0.0
      overall_time[N] = 0.0

      for r in range(runs):

        if N == mem_history_N and r == 0 and device == "cuda":
            torch.cuda.memory._record_memory_history()

        overall_start = time.time()

        # Initialize tensors
        try:
            A = torch.rand(N, N, dtype=dt, device=device)
        except Exception as e:
            print("Failed to allocate tensor A")
            print(f"Error: {e}")
            if N == mem_history_N and r == 0 and device == "cuda":
                torch.cuda.memory._dump_snapshot("malloc_snapshot.pickle")
            sys.exit(0)

        try:
            X = torch.rand(N, second_dimension, dtype=dt, device=device)
        except Exception as e:
            print("Failed to allocate tensor X")
            print(f"Error: {e}")
            if N == mem_history_N and r == 0 and device == "cuda":
                torch.cuda.memory._dump_snapshot("malloc_snapshot.pickle")
            sys.exit(0)

        try:
            Y = torch.rand(N, second_dimension, dtype=dt, device=device)
        except Exception as e:
            print("Failed to allocate tensor Y")
            print(f"Error: {e}")
            if N == mem_history_N and r == 0 and device == "cuda":
                torch.cuda.memory._dump_snapshot("malloc_snapshot.pickle")
            sys.exit(0)

        try:
            B = torch.empty(N, second_dimension, dtype=dt, device=device)
        except Exception as e:
            print("Failed to allocate tensor B")
            print(f"Error: {e}")
            if N == mem_history_N and r == 0 and device == "cuda":
                torch.cuda.memory._dump_snapshot("malloc_snapshot.pickle")
            sys.exit(0)

        # Measure execution time
        mm_start = time.time()
        try:
            B = torch.matmul(A, X) + Y    # Perform matrix-vector multiplication and addition
        except Exception as e:
            print("Failed torch.matmul()")
            print(f"Error: {e}")
            sys.exit(0)
        mm_end = time.time()

        mm_time = mm_end - mm_start
        results[N] = results[N] + mm_time
        overall_time[N] = overall_time[N] + (mm_end - overall_start)
        if args.verbose:
            print(f" Completed run {r}, mm_time was {mm_time:9.6f} ")

        del A, X, Y, B
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

      results[N] = results[N] / (runs)
      overall_time[N] = overall_time[N] / (runs)

      print(f"N={N:6}, Matmul Execution Time: {results[N]:9.6f} seconds \t\t  Overall  Execution Time: {overall_time[N]:9.6f} seconds")

    return results

def plot_results(results):
    plt.figure(figsize=(10, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
    plt.xlabel('Vector Size (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Vector Size')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Basic torch.matmul() test example A*X + Y where A is a 2-d tensor, X, Y can be 2-d or 1-d')
    parser.add_argument('-d', '--device', type=str, help='Specify a device (cpu or cuda)', default = "cpu")
    parser.add_argument('-t', '--dtype', type=str, help='Specify elements type (bfloat16, float16 or float32)', default = "float32")
    parser.add_argument('-r', '--runs', type=int, help='Specify number of runs', default = 1)
    parser.add_argument('-m', '--matrix_mode', action='store_true', help='Enable Matrix-matrix mult vs Matrix-Vector multiplication', default=False)
    parser.add_argument('-y', '--mem_history', type=int, help='Record CUDA memory history during run of size N', default = 0)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose info printing')
    args = parser.parse_args()

    device = args.device
    dtype = args.dtype
    runs = args.runs
    matrix_mode = args.matrix_mode
    mem_history_N = args.mem_history

    if device != "cpu" and device != "cuda":
        device = "cpu"

    match dtype:
        case "bfloat16":
            dt = torch.bfloat16
        case "float32":
            dt = torch.float32
        case "float16":
            dt = torch.float16
        case _:
            dt = torch.float32

    if runs < 1:
      runs = 1

    N_values = [50, 100, 500, 1000, 5000, 10000, 50000]  # Define different N values

    results = matrix_vector_operations(N_values)
    plot_results(results)

