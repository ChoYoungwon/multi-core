#ifndef _TID_INDEX_H
#define _TID_INDEX_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaTid 
{
private:
    int block_tid[3] = {0};
    int grid_tid[3] = {0};

    int tid_in_block;
    int num_thread_in_block;

public:
    // 생성자에서 블록의 차원을 결정
    __device__ CudaTid(int b_dimension = 1) {
        block_tid[0] = threadIdx.x;                                                     // 1차원 x축
        // block_tid[0] = threadIdx.y;                                                  // 1차원 y축
        block_tid[1] = (blockDim.x * threadIdx.y) + block_tid[0];                       // 2차원
        block_tid[2] = ((blockDim.x * blockDim.y) * threadIdx.z) + block_tid[1];        // 3차원

        tid_in_block = block_tid[b_dimension - 1];
        num_thread_in_block = blockDim.x * blockDim.y * blockDim.z;
        grid_tid[0] = (blockIdx.x * num_thread_in_block) + tid_in_block;                // 1차원 x축
        // grid_tid[0] = (blockIdx.y * num_thread_in_block) + tid_in_block;                // 1차원 y축
        grid_tid[1] = blockIdx.y * (gridDim.x * num_thread_in_block) + grid_tid[0];
        grid_tid[2] = blockIdx.z * (gridDim.y * gridDim.x * num_thread_in_block) + grid_tid[1];
    }

    // grid의 차원 결정 및 스레드 id 반환
    __device__ int get_tid_in_grid(int g_dimension = 1) {
        return grid_tid[g_dimension - 1];
    }
};

#endif
