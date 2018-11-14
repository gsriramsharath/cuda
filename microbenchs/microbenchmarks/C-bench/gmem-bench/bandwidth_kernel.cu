struct timing_stats
{
	int start_time;
	int end_time;
};

// Make each cell point to next cell
__global__ void fill_array_kernel(int * data, int requestsize, int size, int stride, int runs)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = (threadNum / requestsize) * stride + threadNum % requestsize;

    
    for(int i = offset; i < size; i += rowSize)
    {
        data[i] = (i + rowSize) % size;
    }
}

__global__ void throughput_stream_pchase_kernel(int * data, int testsize, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
//	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize);
	int nsteps = testsize / threadTotal;

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;

    int value = 0;
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
        value ^= data[j];
    }
    
    int index = offset;
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < nsteps; i++)
    {
        index = data[index];
    }
     end_time = clock();
    __syncthreads();
   
    start_time ^= value;
    start_time ^= value;
    start_time ^= index;
    start_time ^= index;

//	timings[threadNum].start_time = start_time >> 1;
//	timings[threadNum].end_time = end_time >> 1;
}



__device__ void throughput_test_stream(int * data, int runs, int offset, int size, int rowSize, int & value, unsigned int volatile & start_time, unsigned int volatile & end_time)
{
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < runs; ++i)
    {
        for(int j = offset; j < size; j += rowSize)
        {
            value ^= data[j];
        }
    }
    end_time = clock();
    __syncthreads();
    start_time ^= value;
    start_time ^= value;
}

// Specialization
__device__ void throughput_test_stream_1access(int * data, int runs, int offset, int size, int rowSize, int & value, unsigned int volatile & start_time, unsigned int volatile & end_time, int zero)
{
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < runs; ++i)
    {
        value ^= data[offset += zero];
    }
    end_time = clock();
    __syncthreads();
    start_time ^= value;
    start_time ^= value;
}


__global__ void throughput_stream_kernel(int * data, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;
    
    int value = 0;
    
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
        value ^= data[j];
    }
    
    if((size / stride) * requestsize > threadTotal)
        throughput_test_stream(data, runs, offset, size, rowSize, value, start_time, end_time);
    else
        throughput_test_stream_1access(data, runs, offset, size, rowSize, value, start_time, end_time, zero);
    
	timings[threadNum].start_time = start_time >> 1;
	timings[threadNum].end_time = end_time >> 1;
}

__global__ void throughput_texture_pchase_kernel(int * data, int testsize, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);
	int nsteps = testsize / threadTotal;

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;

    int value = 0;
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
         value ^= tex1Dfetch(texRef, j);
    }
    
    int index = offset;

    // partial unrolling
    __syncthreads();
    start_time = clock();
    int i;
    for(i = 0; i < nsteps - 3; i+=4)
    {
        index = tex1Dfetch(texRef, index);
        index = tex1Dfetch(texRef, index);
        index = tex1Dfetch(texRef, index);
        index = tex1Dfetch(texRef, index);
    }
    for(i; i < nsteps; i++)
    {
        index = tex1Dfetch(texRef, index);
    }
     end_time = clock();
    __syncthreads();
   
    start_time ^= value;
    start_time ^= value;
    start_time ^= index;
    start_time ^= index;

	timings[threadNum].start_time = start_time >> 1;
	timings[threadNum].end_time = end_time >> 1;
}


__device__ void throughput_test_texture(int * data, int runs, int offset, int size, int rowSize, int & value, unsigned int volatile & start_time, unsigned int volatile & end_time)
{
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < runs; ++i)
    {
        for(int j = offset; j < size; j += rowSize)
        {
            value ^= tex1Dfetch(texRef, j);
        }
    }
    end_time = clock();
    __syncthreads();
    start_time ^= value;
    start_time ^= value;
}

__device__ void throughput_test_texture_1access(int * data, int runs, int offset, int size, int rowSize, int & value, unsigned int volatile & start_time, unsigned int volatile & end_time, int zero)
{
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < runs; ++i)
    {
        value ^= tex1Dfetch(texRef, offset += zero);
    }
    end_time = clock();
    __syncthreads();
    start_time ^= value;
    start_time ^= value;
}

__global__ void throughput_texture_kernel(int * data, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = threadIdx.y * blockDim.x + threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x * blockDim.y + threadNumInBlock;
	int threadTotal = blockDim.x * blockDim.y * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;
    
    int value = 0;
    
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
        value ^= tex1Dfetch(texRef, j);
    }

    if((size / stride) * requestsize > threadTotal)
        throughput_test_texture(data, runs, offset, size, rowSize, value, start_time, end_time);
    else
        throughput_test_texture_1access(data, runs, offset, size, rowSize, value, start_time, end_time, zero);
   
	timings[threadNum].start_time = start_time >> 1;
	timings[threadNum].end_time = end_time >> 1;
}

#ifdef ATOMICS

__global__ void throughput_atomic_pchase_kernel(int * data, int testsize, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);
	int nsteps = testsize / threadTotal;

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;

    int value = 0;
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
         value ^= data[j];
    }
    
    int index = offset;
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < nsteps; i++)
    {
        index = atomicAdd(&data[index], zero);
    }
     end_time = clock();
    __syncthreads();
   
    start_time ^= value;
    start_time ^= value;
    start_time ^= index;
    start_time ^= index;

	timings[threadNum].start_time = start_time >> 1;
	timings[threadNum].end_time = end_time >> 1;
}

#endif

__global__ void throughput_readwrite_pchase_kernel(int * data, int testsize, int requestsize, int size, int stride, int runs, timing_stats * timings, int zero)
{
	int threadNumInBlock = /*threadIdx.y * blockDim.x +*/ threadIdx.x;
	int threadNum = blockIdx.x * blockDim.x /** blockDim.y*/ + threadNumInBlock;
	int threadTotal = blockDim.x /** blockDim.y*/ * gridDim.x;
	
	int rowSize = (threadTotal / requestsize) * stride;
	int offset = ((threadNum / requestsize) * stride + threadNum % requestsize) % ((size / stride) * requestsize);
	int nsteps = testsize / threadTotal;

    volatile unsigned int start_time = 1664;
    volatile unsigned int end_time = 1664;

    int value = 0;
    // warmup
    for(int j = offset; j < size; j += rowSize)
    {
         value ^= data[j];
    }
    
    int index = offset;
    __syncthreads();
    start_time = clock();
    for(int i = 0; i < nsteps; i++)
    {
        index = data[index] += zero;
    }
     end_time = clock();
    __syncthreads();
   
    start_time ^= value;
    start_time ^= value;
    start_time ^= index;
    start_time ^= index;

	timings[threadNum].start_time = start_time >> 1;
	timings[threadNum].end_time = end_time >> 1;
}

