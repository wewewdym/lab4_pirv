cmake_minimum_required(VERSION 3.18)
project(cuda_lab LANGUAGES C CXX CUDA)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Установка стандарта CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Включаем поддержку CUDA
enable_language(CUDA)

# Пути к заголовочным файлам
include_directories(include)

# Собираем исходные файлы
file(GLOB SRC
    src/*.cpp
    src/*.cu
)

# Создаём исполняемый файл
add_executable(cuda_lab ${SRC})

# Свойства CUDA-цели
set_target_properties(cuda_lab PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 61 70 75 86  # можно адаптировать под вашу карту
)

# Флаг для поддержки расширенных лямбд (нужно для некоторых современных возможностей CUDA)
target_compile_options(cuda_lab PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
