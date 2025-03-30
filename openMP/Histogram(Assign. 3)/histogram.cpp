#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_definitions.h"
#include "DS_timer.h"

#define N (1024 * 1024 * 100)

float random_number(int max)
{
    return rand() % ((max) * 10) / 10.0;
}

const void confirm_result(const int * result1, const int * result2, const int * result3, const int * result4, int _m)
{
    for (int i = 0; i < _m; i++) {
        if (result1[i] != result2[i] || result2[i] != result3[i] || result3[i] != result4[i]) {
            printf("Wrong");
            return;
        }
    }
    printf("Correct");
}

// Serial version
const int * serial_version(const float * _Data, int _m)
{
    int * Bin = (int *)malloc(sizeof(int) * _m);
    memset(Bin, 0, sizeof(int) * _m);
    int i =0;
    for (i = 0; i < N; i++) {
        Bin[((int)_Data[i])]++;
    }

    // printf("Serial    : ");
    // for (int i = 0; i < _m; i++) {
    //     printf("%d ", Bin[i]);
    // }

    // printf("\n");
    return Bin;
}

const int * version_1(const float * _Data, int _m, int _thread_num)
{
    int * Bin = (int *)malloc(sizeof(int) * _m);
    memset(Bin, 0, sizeof(int) * _m);

    #pragma omp parallel num_threads(_thread_num) 
    {
        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            #pragma omp atomic
            Bin[(int)_Data[i]] += 1;
        }
    }

    // printf("Version 1 : ");
    // for (int i = 0; i < _m; i++) {
    //     printf("%d ", Bin[i]);
    // }

    // printf("\n");
    return Bin;
}

const int * version_2(const float * _Data, int _m, int _thread_num)
{
    int * Bin = (int *)malloc(sizeof(int) * _m);
    memset(Bin, 0, sizeof(int) * _m);

    int ** LocalBins = (int **)malloc(sizeof(int *) * _thread_num);
    for (int i = 0; i < _thread_num; i++)
    {
        LocalBins[i] = (int *)malloc(sizeof(int)*_m);
        memset(LocalBins[i], 0, sizeof(int) * _m);
    }

    #pragma omp parallel num_threads(_thread_num)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            LocalBins[tid][(int)_Data[i]]++;
        }

        #pragma omp barrier

        #pragma omp for
        for (int i = 0; i < _thread_num; i++) {
            for (int j = 0; j < _m; j++)
            {
                #pragma omp atomic
                Bin[j] += LocalBins[i][j];
            }
        }
    }
    
    for (int i = 0; i < _thread_num; i++)
        free(LocalBins[i]);
    free(LocalBins);

    return Bin;
}

const int * version_3(const float * _Data, int _m, int _thread_num)
{
    int * Bin = (int *)malloc(sizeof(int) * _m);
    memset(Bin, 0, sizeof(int) * _m);

    int ** LocalBins = (int **)malloc(sizeof(int *) * _thread_num);
    for (int i = 0; i < _thread_num; i++)
    {
        LocalBins[i] = (int *)malloc(sizeof(int)*_m);
        memset(LocalBins[i], 0, sizeof(int) * _m);
    }

    #pragma omp parallel num_threads(_thread_num)
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            LocalBins[tid][(int)_Data[i]]++;
        }
    }

    int step = _thread_num;
    while(step > 1) {
        if (step % 2 == 0) {
            #pragma omp parallel for num_threads(_thread_num)
            for (int i = 0; i < step / 2; i++) {
                for (int j = 0; j < _m; j++)
                LocalBins[i][j] += LocalBins[i + step / 2][j];
            }
            step /= 2;
        } else {
            #pragma omp parallel for num_threads(_thread_num)
            for (int i = 0; i < _m; i++) {
                LocalBins[0][i] += LocalBins[step-1][i];
            }
            step--;
        }
    }   

    memcpy(Bin, LocalBins[0], sizeof(int) * _m);
    
    for (int i = 0; i < _thread_num; i++)
        free(LocalBins[i]);
    free(LocalBins);

    return Bin;
}

int main(int argc, char ** argv)
{
    if (argc < 3) {
        printf("Usage: %s <m> <num_threads>\n", argv[0]);
        return 1;
    }
    
    DS_timer timer(4);
    timer.setTimerName(0, (char*)"[Serial]");
    timer.setTimerName(1, (char*)"[Version_1]");
    timer.setTimerName(2, (char*)"[Version_2]");
    timer.setTimerName(3, (char*)"[Version_3]");


    int m = atoi(argv[1]);
    int thread_num = atoi(argv[2]);

    float * Data = (float *)malloc(sizeof(float) * N);
    memset(Data, 0.0, sizeof(float) * N);
    srand(time(NULL));

    // Data array
    for (int i = 0; i < N; i++) {
        Data[i] = random_number(m);
        // printf("%.1f ", Data[i]);
    }

    // Serial version
    timer.onTimer(0);
    const int * result_serial = serial_version(Data, m);
    timer.offTimer(0);

    // Version 1
    timer.onTimer(1);
    const int * result_version_1 = version_1(Data, m, thread_num);
    timer.offTimer(1);

    // Version 2
    timer.onTimer(2);
    const int * result_version_2 = version_2(Data, m, thread_num);
    timer.offTimer(2);

    // Version 3
    timer.onTimer(3);
    const int * result_version_3 = version_3(Data, m, thread_num);
    timer.offTimer(3);
    
    // confirm coherence
    confirm_result(result_serial, result_version_1, result_version_2, result_version_3, m);


    timer.printTimer();

    free(Data);
    free((void *)result_serial);
    free((void *)result_version_1);
    free((void *)result_version_2);
    free((void*)result_version_3);

    return 0;
}

