#include "brightness.hpp"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr int DELTA = 50;
constexpr int IMAGE_SIZE = WIDTH * HEIGHT;

// CPU: Увеличение яркости
void increaseBrightnessCPU(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    int total = width * height;
    for (int i = 0; i < total; ++i) {
        int pixel = input[i] + delta;
        output[i] = (pixel > 255) ? 255 : pixel;
    }
}

// GPU: Увеличение яркости
__global__ void increaseBrightnessCUDA(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        int pixel = input[idx] + delta;
        output[idx] = (pixel > 255) ? 255 : pixel;
    }
}

void runBrightnessIncrease() {
    // Выделение памяти на хосте
    auto* input = new unsigned char[IMAGE_SIZE];
    auto* output_cpu = new unsigned char[IMAGE_SIZE];
    auto* output_gpu = new unsigned char[IMAGE_SIZE];

    // Инициализация случайных данных
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        input[i] = static_cast<unsigned char>(rand() % 256);
    }

    // CPU: Измерение времени
    auto start_cpu = std::chrono::high_resolution_clock::now();
    increaseBrightnessCPU(input, output_cpu, WIDTH, HEIGHT, DELTA);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // GPU: Выделение памяти на устройстве
    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, IMAGE_SIZE);
    cudaMalloc(&d_output, IMAGE_SIZE);

    // Копирование входных данных на устройство
    cudaMemcpy(d_input, input, IMAGE_SIZE, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    // GPU: Запуск ядра и измерение времени
    auto start_gpu = std::chrono::high_resolution_clock::now();
    increaseBrightnessCUDA<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT, DELTA);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Копирование результатов с устройства на хост
    cudaMemcpy(output_gpu, d_output, IMAGE_SIZE, cudaMemcpyDeviceToHost);

    // Проверка на корректность
    bool success = true;
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        if (output_cpu[i] != output_gpu[i]) {
            std::cerr << "Ошибка в пикселе " << i << ": CPU=" << static_cast<int>(output_cpu[i])
                      << ", GPU=" << static_cast<int>(output_gpu[i]) << '\n';
            success = false;
            break;
        }
    }

    // Вывод времени
    auto cpu_time = std::chrono::duration<float>(end_cpu - start_cpu).count();
    auto gpu_time = std::chrono::duration<float>(end_gpu - start_gpu).count();

    std::cout << "CPU Time: " << cpu_time << " s\n";
    std::cout << "GPU Time: " << gpu_time << " s\n";
    std::cout << (success ? "Brightness increase successful.\n" : "Mismatch in results!\n");

    // Освобождение ресурсов
    delete[] input;
    delete[] output_cpu;
    delete[] output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
}
