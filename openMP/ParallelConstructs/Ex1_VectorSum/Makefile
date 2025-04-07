CXX = g++
CXXFLAGS = -g -Wall -O2 -fopenmp
TARGET = VectorSum
SRCS = VectorSum.cpp DS_timer.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
