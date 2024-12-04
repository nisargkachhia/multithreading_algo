import numpy as np
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# Seed random generator uniquely for every run
np.random.seed(int(time.time()))

def matrix_multiply_worker(args):
    """Worker function to compute a chunk of rows of the result matrix."""
    A_chunk, B = args
    return np.dot(A_chunk, B)

def parallel_matrix_multiply(A, B, num_threads):
    """
    Multiplies two matrices A and B in parallel using the specified number of threads.
    """
    n = len(A)
    chunk_size = n // num_threads

    # Split A into evenly sized chunks
    chunks = [A[i * chunk_size: (i + 1) * chunk_size] for i in range(num_threads)]

    # Handle leftover rows
    if n % num_threads != 0:
        chunks.append(A[num_threads * chunk_size:])

    # Use multiprocessing pool to compute chunks in parallel
    with Pool(num_threads) as pool:
        results = pool.map(matrix_multiply_worker, [(chunk, B) for chunk in chunks])

    # Combine results from all threads
    return np.vstack(results)

def serial_matrix_multiply(A, B):
    """Multiplies two matrices A and B serially."""
    return np.dot(A, B)

def test_correctness(matrix_size, num_threads):
    """Tests the correctness of the parallel matrix multiplication implementation."""
    print(f"\nTesting correctness for matrix size {matrix_size}x{matrix_size} with {num_threads} threads...")
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    serial_result = serial_matrix_multiply(A, B)
    parallel_result = parallel_matrix_multiply(A, B, num_threads)

    # Ensure results match with a tolerance for floating-point precision
    assert np.allclose(serial_result, parallel_result, atol=1e-8), "Results do not match!"
    print("Correctness test passed.")

def benchmark_matrix_multiply(matrix_sizes, num_threads):
    """
    Benchmarks serial vs. parallel matrix multiplication and generates runtime graphs.
    """
    print(f"\nUsing {num_threads} threads for parallel computation.\n")
    serial_times = []
    parallel_times = []

    for size in matrix_sizes:
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        print(f"Matrix size: {size}x{size}")

        # Serial execution
        start_time = time.perf_counter()
        serial_result = serial_matrix_multiply(A, B)
        serial_time = time.perf_counter() - start_time
        serial_times.append(serial_time)
        print(f"  Serial time: {serial_time:.4f} seconds")

        # Parallel execution
        start_time = time.perf_counter()
        parallel_result = parallel_matrix_multiply(A, B, num_threads=num_threads)
        parallel_time = time.perf_counter() - start_time
        parallel_times.append(parallel_time)
        print(f"  Parallel time: {parallel_time:.4f} seconds")

        # Speedup calculation
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        print("-" * 40)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, serial_times, label="Serial Execution", marker='o')
    plt.plot(matrix_sizes, parallel_times, label=f"Parallel Execution ({num_threads} threads)", marker='o')
    plt.title("Runtime Comparison: Serial vs Parallel Matrix Multiplication")
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"runtime_comparison_{num_threads}_threads_{matrix_sizes[0]}_{matrix_sizes[-1]}.png")
    plt.show()

if __name__ == "__main__":
    print("Welcome to the Matrix Multiplication Benchmark Tool!")
    matrix_sizes = list(map(int, input("Enter matrix sizes to test (comma-separated, e.g., 500,1000,1500): ").split(',')))
    num_threads = int(input(f"Enter the number of threads to use (1-{cpu_count()}): "))

    print("\nStarting tests...")
    # Run correctness test for the smallest matrix size
    test_correctness(matrix_size=matrix_sizes[0], num_threads=num_threads)

    # Run benchmark and generate graph
    benchmark_matrix_multiply(matrix_sizes=matrix_sizes, num_threads=num_threads)
