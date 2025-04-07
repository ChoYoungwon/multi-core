#include <stdio.h>
#include <omp.h>

int main(void)
{
    int tid = -1;
    int priVar = -1;

    #pragma omp parallel for num_threads(4) firstprivate(priVar) lastprivate(tid, priVar)
    for(int i = 0; i < 4; i++) {
       tid = omp_get_thread_num();
       priVar = priVar * tid * 10;
       printf("[Thread %d] priVar = %d\n", tid, priVar);
    }

    printf("After parallel region: tid = %d, priVar = %d\n", tid, priVar);
}