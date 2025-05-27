#include "vector_add.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

constexpr int N = 1'000'000;
constexpr int BLOCK_SIZE = 256;
constexpr float EPSILON = 1e-5f;

// CPU-реализация сложения векторов
void vectorAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// GPU-реализация с использованием CUDA
__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void runVectorAddition() {
    size_t bytes = N * sizeof(float);

    // Выделение и инициализация массивов на хосте
    float *A = new float[N];
    float *B = new float[N];
    float *C_cpu = new float[N];
    float *C_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Замер времени выполнения на CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Копирование данных на устройство
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Запуск ядра и замер времени выполнения на GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddCUDA<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Копирование результатов обратно на хост
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Сравнение результатов
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(C_cpu[i] - C_gpu[i]) > EPSILON) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": CPU = " << C_cpu[i] 
                      << ", GPU = " << C_gpu[i] << '\n';
            break;
        }
    }

    // Вывод времени выполнения
    std::chrono::duration<float> cpu_time = end_cpu - start_cpu;
    std::chrono::duration<float> gpu_time = end_gpu - start_gpu;

    std::cout << "CPU Time: " << cpu_time.count() << " s\n";
    std::cout << "GPU Time: " << gpu_time.count() << " s\n";
    std::cout << (correct ? "Results are correct.\n" : "Mismatch in results!\n");

    // Очистка ресурсов
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
