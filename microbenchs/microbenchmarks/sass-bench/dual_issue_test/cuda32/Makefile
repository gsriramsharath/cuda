TARGET = dual_issue
OBJS = dual_issue.o
CUBINS = 

CC = g++

.PRECIOUS: $(OBJS)

all : $(TARGET) 

$(OBJS) : %.o:%.c dual_issue_kernel.cubin
	$(CC) -O2 -c $*.c -o $@ -I/usr/local/cuda/include

$(TARGET):  $(OBJS)
	$(CC) $^ -o $@ -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda 

#$(TARGET):  $(OBJS)
#	ar -rc $@ $^

clean:
	rm -f $(TARGET) $(OBJS)  
