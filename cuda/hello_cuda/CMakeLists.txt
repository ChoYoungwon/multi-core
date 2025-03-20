cmake_minimum_required(VERSION 3.20)

project(hello_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(hello_cuda hello_cuda.cu)

# C++17 및 CUDA 17 지원 설정
target_compile_features(hello_cuda PRIVATE cxx_std_17 cuda_std_17)

# CUDA 코드 분할 컴파일 활성화
set_target_properties(hello_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 라이브러리 및 include 디렉토리 추가
target_include_directories(hello_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# CUDA 런타임 라이브러리 링크
target_link_libraries(hello_cuda PUBLIC CUDA::cudart)