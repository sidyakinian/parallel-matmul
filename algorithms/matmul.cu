#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include "utils/profiler.h"

using std::cout;
using std::endl;
using std::generate;
using std::vector;

const int SHMEM_SIZE = 1 << 10;

std::mutex matmul_mutex;

// Single-threaded naive matrix multiplication on a CPU
// Simplest matrix multiplication algorithm
// Baseline for other algorithms
void baseline_matmul(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

// Tiled matrix multiplication algorithm
// Exploits cache spatial locality
// Is cache-oblivious (we don't need to know the exact size of CPU caches)
void tiled_matmul(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
    int T = 4;
    for (int m = 0; m < N; m += T) {
        for (int n = 0; n < N; n += T) {
            for (int k = 0; k < N; k += T) {
                for (int mt = m; mt < m + T && mt < N; mt++) {
                    for (int nt = n; nt < n + T && nt < N; nt++) {
                        for (int kt = k; kt < k + T && kt < N; kt++) {
                            c[mt * N + nt] += a[mt * N + kt] * b[kt * N + nt];
                        }
                    }
                }
            }
        }
    }
}

// Helper for multithreaded matrix multiplication
void matmul_in_range(vector<int> &a, vector<int> &b, vector<int> &c, int N, int range_start, int range_end) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = range_start; k < range_end; k++) {
                matmul_mutex.lock();
                c[i * N + j] += a[i * N + k] * b[k * N + j];
                matmul_mutex.unlock();
            }
        }
    }
}

// Multithreaded matrix multiiplication algorithm on a CPU
// Uses mutexes to prevent race conditions
// Uses 8 threads by default, the optimal number might vary depending on the number of CPU cores and how many threads each core can process
void multithreaded_matmul(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
    constexpr int num_threads = 8;
    std::vector<std::thread> threads(num_threads);

    for (int i=0; i < num_threads; i++) {
        int range_start = i * N / num_threads;
        int range_end = (i + 1) * N / num_threads;
        std::thread thread(matmul_in_range, std::ref(a), std::ref(b), std::ref(c), N, range_start, range_end);
        threads[i] = std::move(thread);
        // threads[i].detach(); // Detached threads don't need to be joined
    }
    for (int i=0; i < num_threads; i++) {
        threads[i].join();
    }
}

// CUDA kernel for matrix multiplication
__global__ void cuda_matrix_mul(const int *a, const int *b, int *c, int N) {
    // Compute row and column index for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize element of c to zero
    c[row * N + col] = 0;

    // Perform matrix multiplication for a single element of c
    for (int k = 0; k < N; k++) {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

// CUDA kernel for tiled matrix multiplication
__global__ void cuda_tiled_matrix_mul(const int *a, const int *b, int *c, int N) {
    // Compute row and column index for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < N; i += blockDim.x) {
        // Load in elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        // Wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new ones
        __syncthreads();
    }

    // Write back results
    c[row * N + col] = tmp;
}

void naive_mamtul_on_gpu(vector<int> &h_a, vector<int> &h_b, vector<int> &h_c, int N) {
    // Allocate device memory
    size_t bytes = N * N * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    cuda_matrix_mul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy data from device to host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/// Tiled matrix multipl
void tiled_mamtul_on_gpu(vector<int> &h_a, vector<int> &h_b, vector<int> &h_c, int N) {
    // Allocate device memory
    size_t bytes = N * N * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    cuda_tiled_matrix_mul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy data from device to host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // Matrix size of 1024 x 1024;
    int N = 1 << 10;

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // Perform matrix multiplication on CPU/GPU
    auto timed_matmul = make_decorator(tiled_mamtul_on_gpu);
    timed_matmul(h_a, h_b, h_c, N);

    return 0;
}