#!/bin/bash

# 컴파일
g++ -fopenmp -o histogram histogram.cpp DS_timer.cpp -lm

# 테스트할 매개변수 목록
M_VALUES=(10 50 100 500 1000)    # m의 개수 (히스토그램 빈의 수)
THREAD_VALUES=(2 4 6 8)          # 스레드 수

# 결과 파일
RESULT_FILE="test_results.csv"

# CSV 헤더 
echo "M,Threads,Serial(ms),Version1(ms),Version2(ms),Version3(ms),V1_Speedup,V2_Speedup,V3_Speedup" > $RESULT_FILE

for M in "${M_VALUES[@]}"; do
    for THREADS in "${THREAD_VALUES[@]}"; do
        echo "테스트 중: M=$M, Threads=$THREADS"
        
        # 프로그램 실행 및 결과 캡처
        OUTPUT=$(./histogram $M $THREADS)
        
        # 결과가 정확한지 확인
        if [[ $OUTPUT == *"Correct"* ]]; then
            # 시간 결과 추출 - 출력 형식에 맞게 수정
            SERIAL_TIME=$(echo "$OUTPUT" | grep "\[Serial\]" | awk -F': ' '{print $2}' | awk '{print $1}')
            VER1_TIME=$(echo "$OUTPUT" | grep "\[Version_1\]" | awk -F': ' '{print $2}' | awk '{print $1}')
            VER2_TIME=$(echo "$OUTPUT" | grep "\[Version_2\]" | awk -F': ' '{print $2}' | awk '{print $1}')
            VER3_TIME=$(echo "$OUTPUT" | grep "\[Version_3\]" | awk -F': ' '{print $2}' | awk '{print $1}')
            
            # 속도 향상 계산
            V1_SPEEDUP=$(echo "scale=2; $SERIAL_TIME/$VER1_TIME" | bc)
            V2_SPEEDUP=$(echo "scale=2; $SERIAL_TIME/$VER2_TIME" | bc)
            V3_SPEEDUP=$(echo "scale=2; $SERIAL_TIME/$VER3_TIME" | bc)
            
            # CSV에 결과 추가
            echo "$M,$THREADS,$SERIAL_TIME,$VER1_TIME,$VER2_TIME,$VER3_TIME,$V1_SPEEDUP,$V2_SPEEDUP,$V3_SPEEDUP" >> $RESULT_FILE
        else
            echo "오류: M=$M, Threads=$THREADS 조합에서 결과가 일치하지 않습니다."
            echo "$M,$THREADS,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> $RESULT_FILE
        fi
    done
done

echo "모든 테스트 완료. 결과는 $RESULT_FILE 파일에 저장되었습니다."
