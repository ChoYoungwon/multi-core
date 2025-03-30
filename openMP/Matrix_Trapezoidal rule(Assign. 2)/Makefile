cc = gcc
CXX = g++
CFLAGS = -Wall -fopenmp -g
CXXFLAGS = -Wall -fopenmp -g

# 실행파일 이름
EXEC1 = Lab2-1
EXEC2 = Lab2-2

# 소스 파일 
SRC1 = lab1.cpp DS_timer.cpp
SRC2 = lab2.cpp DS_timer.cpp

# 오브젝트 파일
OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)

# 기본 실행 파일 빌드
all: $(EXEC1) $(EXEC2)

# 실행 파일 1 빌드
$(EXEC1): $(OBJ1)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 실행 파일 2 빌드
$(EXEC2): $(OBJ2)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 개별 오브젝트 파일 빌드 규칙
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.rcpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# 정리
clean:
	rm -f $(OBJ1) $(OBJ2) $(EXEC1) $(EXEC2)