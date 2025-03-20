#include <stdio.h>
#include <omp.h>

int main(void)
{
    #pragma omp parallel num_threads(2)
    {
        printf("Thread %d is ready \n", omp_get_thread_num());
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("Section A is executed by thread %d\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                printf("Section B is executed by thread %d\n", omp_get_thread_num());
            }
        }
    }
}