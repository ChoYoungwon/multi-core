#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "DS_timer.h"
#include "DS_definitions.h"

int i, tmp;

double function(double x) { return x * x; }

int main(int argc, char **argv) {

    DS_timer timer(2);
    timer.setTimerName(0, (char*)"[Serial]");
    timer.setTimerName(1, (char*)"[Parallel]");

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int n = atoi(argv[3]);
    double h = (b - a) / n;

    printf("range : (%lf,  %lf), n : %d \n", a, b, n);

    timer.onTimer(0);
    // 직렬 처리
    double serial_sum = 0;
    for (i = 0; i < n; i++) {
        serial_sum += (function(a + h * i) + function(a + h * (i + 1))) / 2.0 * h;
    }
    printf("[Serial] area : %lf\n", serial_sum);

    timer.offTimer(0);


    timer.onTimer(1);
    // 병렬 처리
    double *parallel_sum = (double *)malloc(n * sizeof(double));
    double result = 0.0;
    int thread_num = 8;
    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            parallel_sum[i] = (function(a + h * i) + function(a + h * (i + 1))) / 2 * h;
        }

        
        // 병렬 합 개선 2
        int range = 1;
        while (range < n) {
            #pragma omp for
            for (i = 0; i < n - range; i += 2 * range) {
                parallel_sum[i] += parallel_sum[i + range];
            }
            range *= 2;
        }

        #pragma omp single
        result = parallel_sum[0];
    }

    // 병렬 합합 개선 1
    // double *parallel_tmp = (double *)malloc((n / 2 + 1) * sizeof(double));
    // memset(parallel_tmp, 0, (n / 2 + 1) * sizeof(double));
    // tmp = n;

    // while (n > 1) {
    //     // memset(parallel_tmp, 0, (n / 2 + 1) * sizeof(double));

    //     #pragma omp parallel num_threads(thread_num)
    //     {
    //         #pragma omp for
    //         for (i = 0; i < n / 2; i++) {
    //             parallel_tmp[i] = parallel_sum[i] + parallel_sum[i + n / 2];
    //         }
    //     }

    //     // 마지막 남은 요소 처리
    //     if (n % 2 == 1) {
    //         parallel_tmp[n / 2 - 1] += parallel_sum[n - 1]; 
    //     }

    //     memcpy(parallel_sum, parallel_tmp, (n / 2) * sizeof(double));

    //     n = n / 2;
    // }
    // result = parallel_sum[0];


    printf("[Parallel] area = %lf\n", result);

    timer.offTimer(1);

    timer.printTimer();

    // 메모리 해제
    free(parallel_sum);
    // free(parallel_tmp);

    return 0;
}
