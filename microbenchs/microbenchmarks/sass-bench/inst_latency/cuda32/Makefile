TARGET = inst_latency 
OBJS = inst_latency.o
CUBINS = 

CC = g++

.PRECIOUS: $(OBJS)

all : $(TARGET) 

$(OBJS) : %.o:%.c inst_latency_kernel.cubin
	$(CC) -O2 -c $< -o $@ -I/usr/local/cuda/include


$(TARGET):  $(OBJS)
	$(CC) $^ -o $@ -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda 

#$(TARGET):  $(OBJS)
#	ar -rc $@ $^

clean:
	rm -f $(TARGET) $(OBJS)  
