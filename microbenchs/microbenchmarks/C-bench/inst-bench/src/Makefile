CUDA_INSTALL=/usr/local/cuda
OBJS=main.o empty.o clock.o pipeline.o
TARGET=inst_pipeline

all: $(TARGET)

$(TARGET): $(OBJS)
	g++ $^ -o $@ -L$(CUDA_INSTALL)/lib64 -lcuda -lcudart
#	mkdir -p ../bin; cp -v $@ ../bin/

main.o: main.cpp
	g++ -c $< -o $@ -I$(CUDA_INSTALL)/include

%.o: %.cu instructions.h repeat.h
	nvcc -arch sm_13 -c $< -o $@ 

clean:
	rm -f $(TARGET) *.o
