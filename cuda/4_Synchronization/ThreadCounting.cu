#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void threadCounting_ver1(int * a) {
	atomicAdd(a, 1);
}

__global__ void threadCounting_ver2(int *a) {
	__shared__ int sa;

	if (threadIdx.x == 0)
		sa = 0;
	__syncthreads();

	atomicAdd(&sa, 1);
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(a, sa);
}

int main(void) {
	DS_timer timer(2);
	int a = 0; int *d;

	cudaMalloc((void **)&d, sizeof(int));
	cudaMemset(d, 0, sizeof(int) * 1);

	timer.onTimer(0);
	threadCounting_ver1<<<10240, 512>>>(d);
	cudaDeviceSynchronize();
	timer.offTimer(0);

	cudaMemcpy(&a, d, sizeof(int), cudaMemcpyDeviceToHost);

	timer.onTimer(1);
	threadCounting_ver2<<<10240, 512>>>(d);
	cudaDeviceSynchronize();
	timer.offTimer(1);

	cudaMemcpy(&a, d, sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d\n", a);
	cudaFree(d);

	timer.printTimer();
}