#include <stdio.h>
#include <omp.h>

int main(void)
{
    int a[4] = {0};
    int b[16] = {0};
    #pragma omp parallel num_threads(4)
    {
        int tID = omp_get_thread_num();
        a[tID] = tID * 10;

        // barrier가 없을 경우 여러 스레드가 a[tid]가 초기화되지 않았지만 해당 공간에 접근하여 0을 b배열에 많이 추가하게 됨 =
        #pragma omp barrier

        #pragma omp for
        for (int i = 0; i < 16; i++)
            b[i] = 2 * a[(i + 1) % 4];
    }
}