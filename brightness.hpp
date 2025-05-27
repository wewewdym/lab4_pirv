#pragma once

// Запуск сравнения CPU и GPU версий (управляющая функция)
void runBrightnessIncrease();

// Увеличение яркости на CPU
void increaseBrightnessCPU(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int delta);

// Увеличение яркости на GPU (CUDA-ядро)
__global__ void increaseBrightnessCUDA(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int delta);
