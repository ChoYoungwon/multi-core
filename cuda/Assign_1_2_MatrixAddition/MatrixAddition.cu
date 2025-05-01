#include "DS_timer.h"
#include "Tid_index.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>

__global__ void vecAdd(int * _a, int * _b, int * _c, int _size) {
    CudaTid tid(1);

    int tID = tid.get_tid(1);
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(int argc, char **argv)
{
    const int ROW = 8192;
    const int COL = 8192;
    const int NUM_DATA = ROW * COL;

    // Set timer
	DS_timer timer(5);
	timer.setTimerName(0, (char*)"CUDA Total");
	timer.setTimerName(1, (char*)"Computation(Kernel)");
	timer.setTimerName(2, (char*)"Data Trans. : Host -> Device");
	timer.setTimerName(3, (char*)"Data Trans. : Device -> Host");
	timer.setTimerName(4, (char*)"MatrixAddition on Host");
	timer.initTimers();
	//timer.timerOff();

    int *a, *b, *c, *h_c;
    int *d_a, *d_b, *d_c;       // Matrix on the device
    int memSize = sizeof(int)* ROW * COL;

    a = new int[ROW * COL];  memset(a, 0, memSize);
    b = new int[ROW * COL];  memset(b, 0, memSize);
    c = new int[ROW * COL];  memset(c, 0, memSize);
    h_c = new int[ROW * COL];  memset(h_c, 0, memSize);

    for (int i = 0; i < ROW*COL; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(4);
    for (int i = 0; i < ROW*COL; i++) {
        h_c[i] = a[i] + b[i];
    }
    timer.offTimer(4);

    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    timer.onTimer(0);

    timer.onTimer(2);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    // 2D grid with 1D, 2D blocks
    // int block_x = pow(2, atoi(argv[1]));
    // int block_y = pow(2, atoi(argv[2]));
    // dim3 dimBlock(block_x, block_y, 1);
    // dim3 dimGrid(ceil((float)COL / block_x), ceil((float)ROW / block_y), 1);
    
    // 1D grid with 1D blocks x
    // int block_x = pow(2, atoi(argv[1]));
    // dim3 dimBlock(block_x, 1, 1);
    // dim3 dimGrid(ceil((float)NUM_DATA / block_x), 1, 1);

    // 1D grid with 1D blocks y
    int block_y = pow(2, atoi(argv[1]));
    dim3 dimBlock(1, block_y, 1);
    dim3 dimGrid(1, ceil((float)NUM_DATA / block_y), 1);

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
	delete[] a; delete[] b; delete[] c; delete[] h_c;

	return 0;
}