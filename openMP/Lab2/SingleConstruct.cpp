#include <stdio.h>
#include <omp.h>

int main(void)
{
    float sum = 0;
    float b[10] = { 0.f };
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
            for (int i = 0;  i< 10; i++)
                b[i] = i;

        // #pragma omp single
        // {
            for (int i = 0; i < 10; i++)
                sum += b[i];
        // }

        #pragma omp for
        for (int i = 0; i <10; i++)
            b[i] = b[i]/sum;
    }

    for (int i = 0; i < 9; i++)
        printf("b[%d] = %f \n", i, b[i]);
}