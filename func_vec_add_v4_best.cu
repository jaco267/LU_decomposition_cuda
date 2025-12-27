#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "func.cuh"
#include <vector>
using std::cout;
using std::endl;
// Kernel definition
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  //* 0~65536 *256 + 0~256
    int stride = blockDim.x * gridDim.x; //* 256* 65536
    for (; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const float *h_A, const float *h_B, float *h_C, int N) {
    float *d_A, *d_B, *d_C;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // 2. Copy host → device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    /*
    so for RTX3060 max blockSize is 1024  but best default is 256  
GPUs have a fixed number of Streaming Multiprocessors (SMs).
Each SM can run only a limited number of threads at the same time (e.g., 2048)
If you launch more threads than fit on the GPU, CUDA schedules them in batches.
So, even if you launch millions of threads, only a fraction run truly in parallel; 
the rest wait until hardware resources free up. CUDA hides this from you.
    */
    // 3. Launch kernel with grid-stride loop
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    gridSize = std::min(gridSize, 65535); // 1D grid limit

    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // 4. Copy result device → host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void run_vec_add(){
    //500000 will pass, nice/ 
    int N = 1 << 20;  // ~1 million elements
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    // Call our cuBLAS-style wrapper
    vectorAdd(A.data(), B.data(), C.data(), N);

    // Check result
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " "; // should all be 3
    }
    std::cout << std::endl;
}