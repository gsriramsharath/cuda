TARGET = smem_latency
OBJS = frame.o
CUBINS = 

CC = g++

.PRECIOUS: $(OBJS)

all : $(TARGET) 

frame.o : frame.c frame.h my_kernel_sass.cubin
	$(CC) -O2 -c $*.c -o $@ -I/usr/local/cuda/include

$(TARGET):  $(OBJS)
	$(CC) $^ -o $@ -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda 

clean:
	rm -f $(TARGET) $(OBJS)  
