#include <stdio.h>
#include <omp.h>

int main(void)
{
    int a = 0;
    int result[4] = {0};

    #pragma omp parallel for num_threads(4) private(a) shared(result)
    for(int i = 0; i < 4; i++) {
        a = 0;
        a = a + i;
        a = a * a;

        result[i] = a;
    }

    // default 사용시 컴파일 오류 발생생 + default(shared)시 범위를 지정하지 않은 변수를 모두 shared로 설정
    // #pragma omp parallel for num_threads(4) default(none) private(a)
    // for(int i = 0; i < 4; i++) {
    //     a = 0;
    //     a = a + i;
    //     a = a * a;

    //     result[i] = a;
    // }

    for (int i = 0; i < 4; i++)
        printf("result[%d]a = %d\n", i, result[i]);

}