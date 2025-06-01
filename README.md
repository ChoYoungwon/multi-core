# multi-core
## 목적
병렬 컴퓨팅 : 하나의 문제를 여러 개의 컴퓨팅 자원으로 활용해서 해결
- **동시에 실행**: 여러 작업이 물리적으로 동시에 실행됨
- **다중 프로세서**: 다수의 CPU나 코어를 사용하여 실제로 동시에 연산 수행
- **성능 향상 목적**: 주로 연산 속도를 높이기 위해 사용
- **예시**: 대량의 데이터 처리, 그래픽 렌더링, 과학적 시뮬레이션

## 개요
### OpenMP
#### 1. `ParallelConstructs` : 병렬처리 영역
#### 2. `WorkSharingConstructs` : 작업 분배 방법
  - Loop construct
  - Sections construct
  - Single construct  
#### 4. `Scope of Variables` : 공유 변수 vs 개인 변수
#### 5. `Synchronization` : 동기화
  - Barrier (Implicit barrier, Explicit barrier)
  - critical
  - atomic
  - master
  - Lock
#### 6. `Scheduling` : 작업 스케줄링
  - 정적 vs 동적, 중앙 vs 분산
  - Uniform distribution(균등 분배)
  - Round-robin
  - Master-slave
  - Work-stealing
#### 7. `Reduction` : 누적 연산

### Cuda
#### 1. `VectorSum` : 백터 합
#### 2. `MatrixMultiplication` : 행렬곱 (블록, 그리드 설정)
#### 3. `SharedMemory` : 공유 메모리로 성능 높이기(행렬곱)
#### 4. `Synchronization` : 동기화
#### 5. `assign` : 레이아웃 설정, 행렬 곱 구현, SharedMemory 

## 출처
한국기술교육 대학교 `25 4-1 멀티코어프로그래밍 (김덕수 교수님) \
CUDA 기반 GPU 병렬 처리 프로그래밍(비제이퍼블릭, 2023.05.25, 저자 김덕수) \
https://product.kyobobook.co.kr/detail/S000202185653 \
git : https://github.com/bluekds/CUDA_Programming
