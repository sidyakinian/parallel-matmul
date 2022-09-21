# Parallel Matmul

## Description

This repo contains 5 matrix multiplication algorithms in C++ and CUDA:

1. Baseline single-threaded matrix multiplication.
2. Tiled single-threaded matrix multiplication.
3. Multithreaded matrix multiplication.
4. CUDA kernel for matrix multiplication.
5. CUDA kernel for tiled matrix multiplication.

The purpose of this repo is to compare their implementation and performance.

## Performance comparison

On input size 1024, the algorithms take the following time to execute (in seconds):

![matmul_speeds](https://user-images.githubusercontent.com/34050187/191627038-b644dc79-6822-490a-afcd-921c33dbd9e9.png)

Clearly multithreaded offers an order-of-magnitude better performance, and tiling offers a 20-30% optimization as well due to spatial locality.
