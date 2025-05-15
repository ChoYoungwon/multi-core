#include <stdio.h>
#include <omp.h>
#include "DS_timer.h"
#include <fenv.h>

// CUDA 오류 검사 매크로
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// 몇 개의 샘플 값만 비교(부동 소수점 오차 확인)
void print_sample_values(double *C1, double *C2, int m, int n) {
    printf("Sample values comparison:\n");
    for (int i = 0; i < min(5, m*n); i++) {
        printf("C[%d] = %lf, C_cuda[%d] = %lf, diff = %e\n", 
               i, C1[i], i, C2[i], fabs(C1[i] - C2[i]));
    }
}

// 일치 여부 확인
static void check_correct(double *_C, double *_C_omp, int _m, int _n){
    int e = 0;
    int mismatch_count = 0;
    double max_diff = 0.0;
    int max_diff_idx = -1;

    for (int row = 0; row < _m; row++) {
        for (int col = 0; col < _n; col++) {
            e = _n * row + col;
            double diff = fabs(_C[e] - _C_omp[e]);

            if (diff > 1e-9) {
                mismatch_count++;
                if (diff > max_diff) {
                    max_diff = diff;
                    max_diff_idx = e;
                }
            }
        }
    }
    if (mismatch_count > 0) {
        printf("not matched: %d mismatches out of %d elements\n", mismatch_count, _m * _n);
        printf("Max difference: %e at index %d\n", max_diff, max_diff_idx);
        print_sample_values(_C, _C_omp, _m, _n);
    } else {
        printf("Success matched\n");
    }
}

// 행렬곱 (cuda 커널)
__global__ void MatrixMultiple(double *_A, double * _B, double *_C, int _n, int _k, int _size) {
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size) {
        int row = tID / _n;
        int col = tID % _n;
        double subtotal = 0.0;
        
        for (int i = 0; i < _k; i++) {
            double a = _A[row * _k + i];
            double b = _B[i * _n + col];
            double product = __dmul_rn(a, b);

            subtotal = __dadd_rn(subtotal, product);
        }

        _C[tID] = subtotal;
    }

}

int main(int argc, char **argv) 
{
    if (argc < 4) {
        printf("Usage : ./MatrixMultiplication m n k\n");
        return -1;
    }
    DS_timer timer(6);
    timer.setTimerName(0, (char*)"Serial Algorithm");
    timer.setTimerName(1, (char*)"OpenMP algorithm");
    timer.setTimerName(2, (char*)"CUDA algorithm(total)");
    timer.setTimerName(3, (char*)" - Kernel");
    timer.setTimerName(4, (char*)" - [Data transfer] host -> device");
    timer.setTimerName(5, (char*)" - [Data transfer] device -> host");

    // 라운딩 모드 설정(가장 가까운 숫자로 반올림 설정)
    // fesetround(FE_TONEAREST);

    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);

    const int sizeA = m * k;
    const int sizeB = k * n;
    const int sizeC = m * n;

    // 1차원 방식 동적 할당
    double * A = new double[sizeA]; memset(A, 0, sizeA * sizeof(double));
    double * B = new double[sizeB]; memset(B, 0, sizeB * sizeof(double));
    double * C = new double[sizeC]; memset(C, 0, sizeC * sizeof(double));
    double * C_omp = new double[sizeC]; memset(C_omp, 0, sizeC * sizeof(double));
    double * C_cuda = new double[sizeC]; memset(C_cuda, 0, sizeC * sizeof(double));

    // 무작위 값 할당
    for (int i = 0; i < sizeA; i++) A[i] = (rand() % 100) + (rand() % 100) / 100.0;
    for (int i = 0; i < sizeB; i++) B[i] = (rand() % 100) + (rand() % 100) / 100.0;

    // 직렬 알고리즘
    timer.onTimer(0);
    double subtotal = 0;
    for(int row = 0; row < m; row++) {
        for(int col = 0; col < n; col++) {
            for(int e = 0; e < k; e++) {
                subtotal += A[row * k + e] * B[e * n + col];
            }
            C[n * row + col] = subtotal;
            subtotal = 0;
        }
    }
    timer.offTimer(0);

    // openMP 알고리즘
    timer.onTimer(1);
    #pragma omp parallel for num_threads(4)
    for(int row = 0; row < m; row++) {
        for(int col = 0; col < n; col++) {
            
            double subtotal = 0.0;
            for(int e = 0; e < k; e++) {
                subtotal += A[row * k + e] * B[e * n + col];
            }
            C_omp[n * row + col] = subtotal;
        }
    }
    timer.offTimer(1);

    check_correct(C, C_omp, m, n);

    double * d_A, * d_B, * d_C;

    cudaMalloc(&d_A, sizeof(double) * sizeA); cudaMemset(d_A, 0, sizeof(double) * sizeA);
    cudaMalloc(&d_B, sizeof(double) * sizeB); cudaMemset(d_B, 0, sizeof(double) * sizeB);
    cudaMalloc(&d_C, sizeof(double) * sizeC); cudaMemset(d_C, 0, sizeof(double) * sizeC);

    timer.onTimer(2);
    timer.onTimer(4);
    cudaMemcpy(d_A, A, sizeof(double) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double) * sizeB, cudaMemcpyHostToDevice);
    timer.offTimer(4);

    dim3 dimGrid(ceil((float)m * n / 100), 1, 1);
    dim3 dimBlock(100, 1, 1);

    timer.onTimer(3);
    MatrixMultiple<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n, k, m * n);
    cudaDeviceSynchronize();
    timer.offTimer(3);
    CHECK_CUDA_ERROR(cudaGetLastError()); // 커널 실행 오류 확인
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 커널 완료 대기

    timer.onTimer(5);
    cudaMemcpy(C_cuda, d_C, sizeof(double) * sizeC, cudaMemcpyDeviceToHost);
    timer.offTimer(5);

    timer.offTimer(2);

    check_correct(C, C_cuda, m, n);

    timer.printTimer();
    
    delete[] A; delete[] B; delete[] C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}