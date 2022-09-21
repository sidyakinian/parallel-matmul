# Parallel Matmul

## Description

This repo contains 5 matrix multiplication algorithms in C++ and CUDA:

1. Baseline single-threaded matrix multiplication.
2. Multithreaded matrix multiplication.
3. Tiled single-threaded matrix multiplication.
4. CUDA kernel for matrix multiplication.
5. CUDA kernel for tiled matrix multiplication.

The purpose of this repo is to compare their implementation and performance.

## Performance comparison

On input size 1024, the algorithms take:

1. 24.349s
2. 4.524s
3. 1.247s
4. 0.183s
5. 0.176s
