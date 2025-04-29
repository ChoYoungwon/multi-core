cmake_minimum_required(VERSION 3.20)

project(1_basic_api LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(device_memory device_memory.cu)

target_compile_features(device_memory PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(device_memory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(device_memory PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(device_memory PUBLIC CUDA::cudart)