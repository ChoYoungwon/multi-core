#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW_SIZE (1024)
#define K_SIZE   (512)
#define COL_SIZE (1024)
#define BLOCK_SIZE (32)

#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];	// m * k
float B[K_SIZE][COL_SIZE];	// k * n

// timer
DS_timer* timer;
void setTimer(void);
#define TIMER_HOST			0
#define TIMER_KERNEL		1
#define TIMER_KERNEL_SH		2
#define TIMER_KERNEL_SH_C	3
#define TIMER_HtoD			4
#define TIMER_DtoH			5
#define NUM_TIMER			(TIMER_DtoH+1)

void genInputMatrices(void);
void check_correct(void);

// output matrix
float hostC[ROW_SIZE][COL_SIZE];	// host result
float deviceC[COL_SIZE][COL_SIZE];	// device result

#define memsetZero(_P,_type,_size) memset(_P, 0, sizeof(_type)*(_size));
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*(_size));

#define kernel_rn(sum, a, b) (__fadd_rn(sum, __fmul_rn(a, b)))

__global__ void matMul_kernel(float* _A, float* _B, float* _C)
{
	// int row = blockDim.x * blockIdx.x + threadIdx.x;
	// int col = blockDim.y * blockIdx.y + threadIdx.y;
	
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row >= ROW_SIZE || col >= COL_SIZE)
		return;

	int index = row * COL_SIZE + col;

	float result = 0;
	for (int k = 0; k < K_SIZE; k++) {
		result = kernel_rn(result, _A[row * K_SIZE + k], _B[col + k * COL_SIZE]);
	}
		
	_C[index] = result;
}


__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C)
{
    // TILE_DIM을 64로 설정, BLOCK_SIZE는 32 유지
    const int TILE_DIM = 128;
    
    // 공유 메모리 배열 선언
    __shared__ float sA[BLOCK_SIZE][TILE_DIM];  // 32x128x4 = 16KB
    __shared__ float sB[TILE_DIM][BLOCK_SIZE];  // 128x32x4 = 16KB
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K_SIZE + TILE_DIM - 1) / TILE_DIM; t++) {
        for (int b = 0; b < (TILE_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE; b++) {
			if (row < ROW_SIZE && t * TILE_DIM + tx < K_SIZE) {
				sA[ty][tx + BLOCK_SIZE * b] = _A[row * K_SIZE + (t * TILE_DIM + tx + BLOCK_SIZE * b)];
			} else {
				sA[ty][tx + BLOCK_SIZE * b] = 0.0f;
			}
			
			if (t * TILE_DIM + ty < K_SIZE && col < COL_SIZE) {
				sB[ty + BLOCK_SIZE * b][tx] = _B[(t * TILE_DIM + ty + BLOCK_SIZE * b) * COL_SIZE + col];
			} else {
				sB[ty + BLOCK_SIZE * b][tx] = 0.0f;
			}
		}
        
        __syncthreads();
        
        if (row < ROW_SIZE && col < COL_SIZE) {
            for (int k = 0; k < TILE_DIM; k++) {
                if (t * TILE_DIM + k < K_SIZE) {
                    sum = kernel_rn(sum, sA[ty][k], sB[k][tx]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // 결과 저장
    if (row < ROW_SIZE && col < COL_SIZE) {
        _C[row * COL_SIZE + col] = sum;
    }
}

int main(void)
{
	timer = NULL;	setTimer();

	float* dA, * dB, * dC;
	dA = dB = dC = NULL;

	memsetZero(A, float, MAT_SIZE_A);	memsetZero(B, float, MAT_SIZE_B);
	memsetZero(hostC, float, MAT_SIZE_C);	memsetZero(deviceC, float, MAT_SIZE_C);

	// device memory allocaiton
	dMemAlloc(dA, float, MAT_SIZE_A);
	dMemAlloc(dB, float, MAT_SIZE_B);
	dMemAlloc(dC, float, MAT_SIZE_C);

	// generate input matrices
	genInputMatrices();

	// Host code
	timer->onTimer(TIMER_HOST);
	for (int r = 0; r < ROW_SIZE; r++)
		for (int c = 0; c < COL_SIZE; c++)
			for (int k = 0; k < K_SIZE; k++)
				hostC[r][c] += A[r][k] * B[k][c];
	timer->offTimer(TIMER_HOST);

	// Copy input matrices : H -> D
	timer->onTimer(TIMER_HtoD);
	cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
	timer->offTimer(TIMER_HtoD);

	dim3 gridDim(ceil((float)ROW_SIZE / BLOCK_SIZE), ceil((float)COL_SIZE / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	timer->onTimer(TIMER_KERNEL);
	matMul_kernel << <gridDim, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL);

	timer->onTimer(TIMER_KERNEL_SH);
	matMul_kernel_shared << <gridDim, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL_SH);

	// Get back result : D -> H
	timer->onTimer(TIMER_DtoH);
	cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
	timer->offTimer(TIMER_DtoH);

	// check the results
	check_correct();

	timer->printTimer();
	if (timer != NULL)
		delete timer;
}

void genInputMatrices(void)
{
	for (int r = 0; r < ROW_SIZE; r++)
		for (int k = 0; k < K_SIZE; k++)
			A[r][k] = rand() % 100 + (rand() % 100) / 100.0;

	for (int k = 0; k < K_SIZE; k++)
		for (int c = 0; c < COL_SIZE; c++)
			B[k][c] = rand() % 100 + (rand() % 100) / 100.0;
}

void setTimer(void)
{
	timer = new DS_timer(NUM_TIMER);

	timer->initTimers();
	timer->setTimerName(TIMER_HOST, (char*)"CPU code");
	timer->setTimerName(TIMER_KERNEL, (char*)"Kernel launch");
	timer->setTimerName(TIMER_KERNEL_SH, (char*)"Kernel launch (shared ver.)");
	timer->setTimerName(TIMER_HtoD, (char*)"[Data transter] host->device");
	timer->setTimerName(TIMER_DtoH, (char*)"[Data transfer] device->host");
}

// 몇 개의 샘플 값만 비교(부동 소수점 오차 확인)
void print_sample_values(float* C1, float* C2) {
    printf("Sample values comparison:\n");
    for (int i = 0; i < min(5, ROW_SIZE*COL_SIZE); i++) {
        printf("hostC[%d] = %lf, C_cuda[%d] = %lf, diff = %e\n", 
               i, C1[i], i, C2[i], fabs(C1[i] - C2[i]));
    }
}

// 일치 여부 확인
void check_correct(){

	float* pHostC = &hostC[0][0];
	float* pDeviceC = &deviceC[0][0];

    int e = 0;
    int mismatch_count = 0;
    double max_diff = 0.0;
    int max_diff_idx = -1;

    for (int row = 0; row < ROW_SIZE; row++) {
        for (int col = 0; col < COL_SIZE; col++) {
            e = COL_SIZE * row + col;
            double diff = fabs(pHostC[e] - pDeviceC[e]);

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
        printf("not matched: %d mismatches out of %d elements\n", mismatch_count, COL_SIZE * ROW_SIZE);
        printf("Max difference: %e at index %d\n", max_diff, max_diff_idx);
        print_sample_values(pHostC, pDeviceC);
    } else {
        printf("Success matched\n");
    }
}