# Parallel Matrix Multiplication Benchmark

## Nisarg Jigneshbhai Kachhia

## 1002265218

This project evaluates the runtime differences between serial and parallel implementations of matrix multiplication in Python. It allows users to test the correctness and efficiency of parallel computations, leveraging multiple threads.

## Features

- Serial Matrix Multiplication: Uses a straightforward implementation with numpy for matrix multiplication.
- Parallel Matrix Multiplication: Divides the workload across multiple threads to utilize multicore processing.
- Benchmarking: Measures and compares runtimes for serial and parallel approaches on different matrix sizes.
- Graphical Analysis: Generates runtime comparison graphs for different matrix sizes.

## Requirements

- Python 3.x
- Required libraries: numpy, matplotlib, multiprocessing
- Install the necessary libraries with: pip install numpy matplotlib

## How to use this project

- For compilation and run, enter "python filename.py"
- Input matrix sizes as a comma-separated list, e.g., 500,1000,1500.
- Specify the number of threads to use for parallel execution (up to your CPU's core count).
- The script will:
    1. Validate the correctness of parallel matrix multiplication.
    2. Benchmark serial and parallel implementations.
    3. Display and save a runtime comparison graph.

## Implementation details

1. Matrix Multiplication Methods
    - Serial Matrix Multiplication:
        1. Uses a single thread to compute the matrix product.
        2. Relies on NumPy for efficient matrix operations.
        3. Serves as the baseline for performance comparison.

    - Parallel Matrix Multiplication:
        1. Divides the rows of the first matrix (A) into smaller chunks.
        2. Each chunk is processed in parallel using multiple threads via Python’s multiprocessing module.
        3. Combines the results from all threads to form the final product.

2. Correctness Validation
    - Ensures the accuracy of the parallel implementation by comparing its output with the serial implementation.
    - Accounts for minor floating-point errors by allowing a small tolerance level.
    - Verifies that both outputs match to confirm correctness.

3. Performance Benchmarking
    - Compares the runtimes of serial and parallel implementations for different matrix sizes.
    - Measures key metrics:
        1. Execution Time: Time taken to compute the matrix product.
        2. Speedup: The ratio of serial execution time to parallel execution time.
    - Demonstrates the efficiency gain from parallelization.

4. Graph Generation
    - Visualizes the runtime comparison as a line graph:
        1. X-axis: Matrix sizes.
        2. Y-axis: Execution times for serial and parallel methods.
    - Highlights the performance improvement achieved with parallel computation.
    - Saves the graph as a PNG file for reference.

5. User Interaction
    - The script prompts the user for:
        1. Matrix sizes to test (e.g., 500x500, 1000x1000).
        2. Number of threads to use for parallel computation.
    - Provides real-time feedback during execution, including runtimes and speedup for each test case.

## Conclusion

- This project demonstrates the computational benefits of parallel matrix multiplication, particularly for large matrix sizes. By leveraging multicore processing, significant speedups can be achieved, making it a useful tool for performance-critical applications.
