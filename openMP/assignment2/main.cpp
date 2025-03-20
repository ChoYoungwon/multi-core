#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

// Set the size of matrix and vector
// matrix A = m by n
// vector b = n by 1
#define m (10000)
#define n (10000)

#define GenFloat (rand() % 100 + ((float)(rand() % 100) / 100.0))
void genRandomInput();

float A[m][n];
float X[n];
float Y_serial[m];
float Y_parallel[m];
int i, j;

int main(int argc, char** argv)
{
	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");

	genRandomInput();


	//** 1. Serial code **//
	timer.onTimer(0);

	for (i = 0; i < m; i++) {
		int sum = 0;
		for (j = 0; j < n; j++) {
			sum += A[i][j] * X[j];
		}
		Y_serial[i] = sum;
		printf("%f ", Y_serial[i]);
	}
	printf("\n");

	timer.offTimer(0);

	//** 2. Parallel code **//
	timer.onTimer(1);

	#pragma omp parallel num_threads(8)
	{
		#pragma omp for
		for (i = 0; i < m; i++) {
			int sum = 0;
			for (j = 0; j < n; j++) {
				sum += A[i][j] * X[j];			
			}
			Y_parallel[i] = sum;
		}
	}

	for (i = 0; i < m; i++) {
		printf("%f ", Y_parallel[i]);
	}
	timer.offTimer(1);

	//** 3. Result checking code **//
	bool isCorrect = true;

	for (i = 0; i < m; i++)
	{
		if (Y_parallel[i] != Y_serial[i])
			isCorrect = false;
	}

	if (isCorrect)
		printf("Results are not matched :(\n");
	else
		printf("Results are matched! :)\n");

	timer.printTimer();
	EXIT_WIHT_KEYPRESS;
}

void genRandomInput(void) {
	// A matrix
	LOOP_INDEX(row, m) {
		LOOP_INDEX(col, n) {
			A[row][col] = GenFloat;
		}
	}

	LOOP_I(n)
		X[i] = GenFloat;

	memset(Y_serial, 1, sizeof(float) * m);
	memset(Y_parallel, 0, sizeof(float) * m);
}