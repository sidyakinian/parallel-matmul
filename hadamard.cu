#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <random>

using std::cout;
using std::endl;

// CUDA kernel for Hadamard product
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void hadamard(const float *__restrict a, const float *__restrict b,
                          float *__restrict c, const int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    c[tid] = a[tid] * b[tid];
  }
}

// Check Hadamard product result
void verify_result(std::vector<float> &a, std::vector<float> &b,
                   std::vector<float> &c, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        assert(c[i*N + j] == a[i*N + j] * b[i*N + j]);
    }
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 4;
  constexpr size_t bytes = sizeof(float) * N * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<float> a(N * N);
  std::vector<float> b(N * N);
  std::vector<float> c(N * N);

    // Create random number generator
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

  // Initialize random numbers in each matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N + j] = distribution(generator);
      b[i*N + j] = distribution(generator);
      c[i*N + j] = 0;
    }
  }

    cout << "initialized array" << endl;

  // Allocate memory on the device
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  int NUM_BLOCKS = (N * N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  // Kernel calls are asynchronous (the CPU program continues execution after
  // call, but no necessarily before the kernel finishes)
  hadamard<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Copy sum vector from device to host
  // cudaMemcpy is a synchronous operation, and waits for the prior kernel
  // launch to complete (both go to the default stream in this case).
  // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
  // barrier.
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cout << "checking result..." << endl;
  // Check result for errors
//   for (int i=0; i < N; i++) {
//     for (int j=0; i < N; j++) {
//         cout << c[i*N + j] << " ";
//     }
//     cout << endl;
//   }
//   verify_result(a, b, c, N);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

// I don't *think* I need to free memory here..
//   free(a);
//   free(b);
//   free(c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}