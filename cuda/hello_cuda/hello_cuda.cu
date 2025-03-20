#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloCUDA(void)
{
    printf("Hello CUDA from GPU!\n");
}

int main(void)
{
    printf("Hello GPU from CPU!\n");
    helloCUDA<<<1,10>>>();

    // GPU 작업이 끝날 때까지 대기
    cudaDeviceSynchronize();
    
    return 0;
}