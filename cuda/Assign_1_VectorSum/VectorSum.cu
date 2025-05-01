#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// #define NUM_DATA 1024 * 1024 * 8

__global__ void vecAdd(int * _a, int * _b, int * _c, int _size) {
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(int argc, char ** argv)
{
    if (argc != 4) {
        printf("usage : ./VectorSum num1 num2 num3\n");
        printf("NUM_DATA = num1 * num2 * num3\n");
        return -1;
    }

    const int NUM_DATA = atoi(argv[1]) * atoi(argv[2]) * atoi(argv[3]);

    // Set timer
	DS_timer timer(5);
	timer.setTimerName(0, (char*)"CUDA Total");
	timer.setTimerName(1, (char*)"Computation(Kernel)");
	timer.setTimerName(2, (char*)"Data Trans. : Host -> Device");
	timer.setTimerName(3, (char*)"Data Trans. : Device -> Host");
	timer.setTimerName(4, (char*)"VectorSum on Host");
	timer.initTimers();
	//timer.timerOff();

    int *a, *b, *c, *h_c;       // Vectors on the host
    int *d_a, *d_b, *d_c;       // Vectors on the device

    int memSize = sizeof(int)*NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

    for(int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++)
        h_c[i] = a[i] + b[i];
    timer.offTimer(4);

    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    timer.onTimer(0);

    timer.onTimer(2);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    dim3 dimGrid(ceil((float)NUM_DATA / 256), 1, 1);
    dim3 dimBlock(256, 1, 1);
    timer.onTimer(1);
    vecAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NUM_DATA);
    timer.offTimer(1);

    timer.onTimer(3);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);

    timer.offTimer(0); timer.printTimer();

    // Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (h_c[i] != c[i]) {
			printf("[%d] The resutls is not matched! (%d, %d)\n"
				, i, h_c[i], c[i]);
			result = false;
		}
	}

    if (result)
		printf("GPU works well!\n");

    // Release device memory
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}