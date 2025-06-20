#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// #ifdef _WIN32
// #include <Windows.h>
// #else 
// #include <unistd.h>
// #define Sleep(x) sleep(x)
// #endif

int getOrder() {
	// Sleep(10);
	return rand() % 4 + 1;
}

int main(void) {

	int order[4] = { 0 };
	#pragma omp parallel num_threads(4)
	{
		int tID = omp_get_thread_num();
		
		// 특정 스레드만 일을 하면 되고, 나머지는 기다려야 할 때
		if (tID == 0) { // master thread
			for (int i = 1; i < 4; i++)
				order[i] = getOrder();
		}

		#pragma omp barrier
        
		if (tID != 0) // slave workers
			printf("Thread-%d cooks %d\n", tID, order[tID]);
	}

	return 0;
}