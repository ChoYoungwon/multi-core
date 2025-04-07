#include <stdio.h>
#include <omp.h>

int main(void)
{
    int a[10] = {0};
    int b[10] = {0};
    #pragma omp parallel num_threads(4) 
    {
        // implicit barrier
        #pragma omp for
        for (int i = 0; i < 10; i++)
            a[i] = i;
        // implicit barrier

        // implicit barrier
        #pragma omp for
        for (int i = 0; i < 9; i++)
            b[i] = 2 * a[( i + 1)];
        // implicit barrier
    }

    for (int i = 0; i < 10; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
}