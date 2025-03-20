#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "DS_timer.h"
#include "DS_definitions.h"

int i, tmp;
float parallel_result = 0;

double function(double x) { return x * x; }

int main(int argc, char **argv) {

    DS_timer timer(2);
    timer.setTimerName(0, (char*)"serial");
    timer.setTimerName(1, (char*)"Parallel");

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int n = atoi(argv[3]);
    double h = (double) (b - a) / n;

    printf("a : %d, b : %d, n : %d, h : %lf \n", a, b, n, h);

    timer.onTimer(0);
    // 직렬 처리
    double serial_sum = 0;
    for (i = 0; i < n; i++) {
        serial_sum += (function(a + h * i) + function(a + h * (i + 1))) / 2.0 * h;
    }
    printf("serial_sum : %f\n", serial_sum);

    timer.offTimer(0);


    timer.onTimer(1);
    // 병렬 처리
    float *parallel_sum = (float *)malloc(n * sizeof(float));
    int thread_num = 8;
    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            parallel_sum[i] = (function(a + h * i) + function(a + h * (i + 1))) / 2 * h;
        }
    }

    // 🔹 메모리 크기 조정: (n / 2 + 1) 크기로 할당
    float *parallel_tmp = (float *)malloc((n / 2 + 1) * sizeof(float));
    tmp = n;

    while (n > 1) {
        memset(parallel_tmp, 0, (n / 2 + 1) * sizeof(float));

        #pragma omp parallel num_threads(thread_num)
        {
            #pragma omp for
            for (i = 0; i < n / 2; i++) {
                parallel_tmp[i] = parallel_sum[i] + parallel_sum[i + n / 2];
            }
        }

        // 🔹 마지막 남은 요소 처리 (오류 수정)
        if (n % 2 == 1) {
            parallel_tmp[n / 2 - 1] += parallel_sum[n - 1];  // ✅ 마지막 요소를 이전 값에 더함
        }

        // parallel_sum을 parallel_tmp로 업데이트
        memcpy(parallel_sum, parallel_tmp, (n / 2) * sizeof(float));

        // 🔹 n 값을 업데이트 (홀수일 경우 고려)
        n = n / 2;
    }

    printf("parallel_sum = %f\n", parallel_sum[0]);

    timer.offTimer(1);

    timer.printTimer();

    // 메모리 해제
    free(parallel_sum);
    free(parallel_tmp);

    return 0;
}
