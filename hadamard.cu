#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for Hadamard product
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void hadamard(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    c[tid] = a[tid] * b[tid];
  }
}

// Check Hadamard product result
void verify_result(std::vector<std::vector<int>> &a, std::vector<std::vector<int>> &b,
                   std::vector<std::vector<int>> &c) {
  for (int i = 0; i < a.size(); i++) {
    for (int j = 0; j < a[i].size(); j++) {
        assert(c[i][j] == a[i][j] * b[i][j]);
    }
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 8;
  constexpr size_t bytes = sizeof(int) * N * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<std::vector<int>> a(N, std::vector<int>(N, 0));
  std::vector<std::vector<int>> b(N, std::vector<int>(N, 0));
  std::vector<std::vector<int>> c(N, std::vector<int>(N, 0));

  std::cout << a[1][1] << std::endl;
  std::cout << b[3][3] << std::endl;

  // Initialize random numbers in each matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] = rand() % 100;
      b[i][j] = rand() % 100;
    }
  }

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
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

  // Check result for errors
//   verify_result(a, b, c);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(a);
  free(b);
  free(c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}