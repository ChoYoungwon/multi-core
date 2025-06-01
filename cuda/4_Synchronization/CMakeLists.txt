# CMake 최소 버전 요구사항 설정 (3.20 이상 필요)
cmake_minimum_required(VERSION 3.20)

# 프로젝트 이름 및 사용 언어 정의 (C++ 및 CUDA)
project(ThreadCounting LANGUAGES CXX CUDA)

# CUDA 툴킷 패키지 찾기 (필수)
find_package(CUDAToolkit REQUIRED)

# OpenMP 패키지 찾기 (필수)
find_package(OpenMP REQUIRED)

# 지원할 CUDA 아키텍처 설정 (여러 세대의 NVIDIA GPU 지원)
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)

# CUDA 컴파일러 플래그 설정 (deprecated GPU 타겟 경고 비활성화)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# C++ 컴파일러에 OpenMP 플래그 추가
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# CUDA 컴파일러에 호스트 컴파일러용 OpenMP 플래그 추가 (-Xcompiler 옵션 사용)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")

# 실행 파일 생성
add_executable(ThreadCounting ThreadCounting.cu DS_timer.cpp)

# C++17 및 CUDA 17 지원 설정
target_compile_features(ThreadCounting PRIVATE cxx_std_17 cuda_std_17)

# CUDA 코드 분할 컴파일 활성화
set_target_properties(ThreadCounting PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 헤더 파일 포함 디렉토리 추가 (CUDA 툴킷 헤더 포함)
target_include_directories(ThreadCounting PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# 필요한 라이브러리 링크 (CUDA 런타임 및 OpenMP 라이브러리)
target_link_libraries(ThreadCounting PUBLIC CUDA::cudart ${OpenMP_CXX_LIBRARIES})