# Add source files here
APP	:= bandwidth

# Cuda source files (compiled with cudacc)
CUFILES		:= bandwidth.cu

CUDA_INSTALL_PATH := /usr/local/cuda

NVCC       := nvcc 

# Libs
LDFLAGS    := -lm #-lcuda

CFLAGS := -arch=sm_20

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g -G
	CFLAGS   += -D_DEBUG
else 
	COMMONFLAGS += -O3 
	CUDACCFLAGS  += --compiler-options -fno-strict-aliasing
endif

# Add cudacc flags
CFLAGS += $(CUDACCFLAGS)

# Add common flags
CFLAGS += $(COMMONFLAGS)

################################################################################
# Set up object files
################################################################################
OBJS :=  $(patsubst %.cu,%.o,$(CUFILES))

all: $(APP)

%.o : %.cu
	$(NVCC) -c $< $(CFLAGS) -o $@
#	$(NVCC) -M $< $(CFLAGS) > $@.dep

#
# Pick up generated dependency files, and
# add /dev/null because gmake does not consider
# an empty list to be a list:
#
include $(wildcard *.dep) /dev/null


$(APP) : $(OBJS)
	$(NVCC) $^ $(LDFLAGS) -o $@


#
# Cleanup:
#
clean :
	$(RM) $(OBJS) *.dep $(APP)


