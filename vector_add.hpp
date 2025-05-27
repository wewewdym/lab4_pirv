#pragma once

// Запуск сравнения CPU и GPU версий сложения векторов
void runVectorAddition();

// Сложение векторов на CPU
void vectorAddCPU(
    const float* A,
    const float* B,
    float* C,
    int N);

// Сложение векторов на GPU (CUDA kernel)
__global__ void vectorAddCUDA(
    const float* A,
    const float* B,
    float* C,
    int N);
