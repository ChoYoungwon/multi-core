#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"

#define f(_x) (_x*_x)
#define NUM_THREADS (4)
#define offset (16)

enum Algorithm {
    Serial, Parallel_offset, Parallel_critical, Parallel_atomic, END
};

double Trapezodial_Serial(double a, double b, int n, double h);
double Trapezodial_Parallel_offset(double a, double b, int n, double h);
double Trapezodial_Parallel_critical(double a, double b, int n, double h);
double Trapezodial_Parallel_atomic(double a, double b, int n, double h);

#define RUN_TEST(_func, a, b, n, h) { \
    timer.onTimer(Algorithm::_func);  \
    sum = Trapezodial_##_func(a, b, n, h); \
    timer.offTimer(Algorithm::_func); \
    printf("[%s] area = %lf (%.2f x)\n", #_func, sum \
        , timer.getTimer_ms(Algorithm::Serial) / timer.getTimer_ms(Algorithm::_func));  \
}

int main(void)
{
    DS_timer timer(Algorithm::END);
    timer.setTimerName(Algorithm::Serial, (char*)"Serial algorithm");
    timer.setTimerName(Algorithm::Parallel_offset, (char*)"Parallel algorithm");
    timer.setTimerName(Algorithm::Parallel_critical, (char*)"Parallel algorithm - critical section");
    timer.setTimerName(Algorithm::Parallel_atomic, (char*)"Parallel algorithm - atomic");
    
    double a = -1, b = 1;
    int n = (1024 * 1024 * 100);

    double h = (b - a) / n;
    printf("f(x) = x * x\n");
    printf("range = (%lf, %lf), n = %d, h = %.12lf\n", a, b, n, h);
    double sum = 0;

    RUN_TEST(Serial, a, b, n, h);
    RUN_TEST(Parallel_offset, a, b, n, h);
    RUN_TEST(Parallel_critical, a, b, n, h);
    RUN_TEST(Parallel_atomic, a, b, n, h);

    timer.printTimer();
}

// 직렬처리리
double Trapezodial_Serial(double a, double b, int n, double h)
{
    double sum = 0;
    for (int i = 0; i < n - 1; i++)
    {
        double x_i = a + h * i;
        double x_j = a + h * (i + 1);
        double d = (f(x_i) + f(x_j)) / 2.0;
        sum += d * h;
    }
    return sum;
}

// 병렬처리 (cache coherece 고려)
double Trapezodial_Parallel_offset(double a, double b, int n, double h)
{
    double sum = 0;
    double local[NUM_THREADS * offset] = {0};
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num() * offset;
        #pragma omp for
        for (int i = 0; i < n - 1; i++)
        {
            double x_i = a + h * i;
            double x_j = a + h * (i + 1);
            double d = (f(x_i) + f(x_j)) / 2.0;
            local[tid] += d * h;
        }
    }
    for (int i = 0; i < NUM_THREADS; i++)
        sum += local[i * offset];
    
    return sum;
}

// critical 구현
double Trapezodial_Parallel_critical(double a, double b, int n, double h)
{
    double sum = 0;
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n - 1; i++)
    {
        double x_i = a + h * i;
        double x_j = a + h * (i + 1);
        double d = (f(x_i) + f(x_j)) / 2.0;
        
        #pragma omp critical
        {
            sum += d * h;
        }
    }
    return sum;
}

// atomic 구현
double Trapezodial_Parallel_atomic(double a, double b, int n, double h)
{
    double sum = 0;
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n - 1; i++)
    {
        double x_i = a + h * i;
        double x_j = a + h * (i + 1);
        double d = (f(x_i) + f(x_j)) / 2.0;
        
        #pragma omp atomic
        sum += d * h;
    }
    return sum;
}