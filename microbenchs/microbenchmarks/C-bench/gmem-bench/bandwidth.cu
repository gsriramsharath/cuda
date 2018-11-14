
#include "utils.h"
#include <stdlib.h>
#include <sys/time.h>
double get_useconds() 
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double) ((tp.tv_sec * (1e+6)) + tp.tv_usec);
}



__global__ void fillmem(int *index,int hwsize,int unit,int mode,int param)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	int nb = tx/hwsize;
	int base  = nb*unit;

	int offset =base ;
	if(mode==0){
		offset += tx%hwsize;
	}
	else if(mode==1){
		offset += 0;// * (unit/hwsize);
	}
	else if(mode==2){
		offset += tx%hwsize + param;
	}

	index[tx]=offset;
}

const int elesize = 16;
const int kernelloop = 10;
struct udata
{
	char s[elesize];
};

__global__ void bandwidth_kernel(unsigned int *time,int *index,char *data,int round,int rowsize,int zero)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	udata ud;
	int pos;	
	
    __syncthreads();
	volatile unsigned int start_time = clock();
#pragma unroll
	for(int j=0;j<kernelloop;j++){
		pos = index[tx];
		for(int i=0;i<round;i++){
			ud = ((udata*)(data+pos))[0];
			pos += rowsize + ((int)(ud.s) & zero);
	//		pos += rowsize ;
		}
	}
	
    __syncthreads();
	volatile unsigned int end_time = clock();
	time[tx] = end_time - start_time;
	
	if(zero > 0)
		index[0]+=pos;
	
}

struct cu16
{
	char s[16];
};
struct cu8
{
	char s[8];
};

template<int mode>
__global__ void memcpy_kernel(char *src,char *dst,int step)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	double2 *d2src=(double2*)src,*d2dst=(double2*)dst;
	double *dsrc = (double*)src, *ddst = (double*)dst;
	switch(mode)
	{
	case 0:
		dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];		
		tx+=step;dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];
		tx+=step;dst[tx] = src[tx];
		break;
	case 1:
		d2dst[tx]=d2src[tx];
		break;
	case 2:
		ddst[tx]=dsrc[tx];
		tx+=step;ddst[tx]=dsrc[tx];
	}
}

__global__ void memcpy_kernel2(double2 *src,double2 *dst)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	dst[tx]=src[tx];
}


void testbandwidth(int zero)
{
	int nMode = 4;
	const int loop = 10;
	unsigned int blocksize = 1024;
	unsigned int gridsize = 15*128*16;
	unsigned int totalmem = 16*gridsize*blocksize+1024;

	cudaDeviceProp props;
	CUDA_SAFE_CALL(cudaSetDevice(0));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, 0));
	unsigned int gmem_total = props.totalGlobalMem;
	if(totalmem*2 > gmem_total){
		fprintf(stderr,"Memory exceed!\n");
		return;
	}
/*	
	unsigned int gridsize_x = 1024;
	unsigned int gridsize_y = totalmem/(gridsize_x*blocksize);
	dim3 gridsize(gridsize_x,gridsize_y);
*/	
	char *src_gpu,*dst_gpu;
	CUDA_SAFE_CALL(cudaMalloc((void**)&src_gpu, totalmem));
	CUDA_SAFE_CALL(cudaMemset(src_gpu, zero, totalmem));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dst_gpu, totalmem));

	printf("gridsize=%d,blocksize=%d\n",gridsize,blocksize);
//	printf("gridsize=(%d,%d),blocksize=%d\n",gridsize_x,gridsize_y,blocksize);

	for(int mode=0;mode<nMode;mode++){
		printf("\nmode#%d\n",mode);
		printf("offset(B)\ttime(ms)\tmemsize(MB)\tbandwidth(GB/s)\n");
		
		for(int offset=0;offset<512;offset+=16){
			char *src = src_gpu+offset;
			char *dst = dst_gpu+offset;
		
			cudaThreadSynchronize();
			double t0 = get_useconds();
			switch(mode){
			case 0:
				for(int i=0;i<loop;i++)
					cudaMemcpy(dst,src,totalmem,cudaMemcpyDeviceToDevice);
				break;
			case 1:
				for(int i=0;i<loop;i++)
					memcpy_kernel<0><<<gridsize,blocksize>>>(src,dst,gridsize*blocksize);
				break;
			case 2:
				for(int i=0;i<loop;i++)
					memcpy_kernel<1><<<gridsize,blocksize>>>(src,dst,gridsize*blocksize);
				break;
			case 3:
				for(int i=0;i<loop;i++)
					memcpy_kernel<2><<<gridsize,blocksize>>>(src,dst,gridsize*blocksize);
				break;
			}
			cudaThreadSynchronize();
			double tt = get_useconds()-t0;
			
			char pos = 0;
			CUDA_SAFE_CALL(cudaMemcpy(&pos,dst,sizeof(char),cudaMemcpyDeviceToHost));
			totalmem+=pos;
			if(pos!=0){
				fprintf(stderr,"Error!\n");
				exit(-1);
			}
			
			printf("%d\t%.2f\t%d\t%.2f\n",offset,tt/(loop*1000),totalmem/(1024*1024),2.0*loop*totalmem/(1000*tt));
		}
	}
	

	CUDA_SAFE_CALL(cudaFree(src_gpu));
	CUDA_SAFE_CALL(cudaFree(dst_gpu));


}

int main(int argc,char *argv[])
{
	testbandwidth(argc>10);
	return 0;
}

int testThroughput(int argc, char ** argv)
{
	const int blocksize = 512;
	const int data_size_unit = 1024;
	const int loop = 3;
	const int nmode = 3;

	cudaDeviceProp props;
   CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, 0));
    size_t gmem_total = props.totalGlobalMem;
    int n_sm = props.multiProcessorCount;
    int n_hwsize = props.warpSize/2;
    int freq = props.clockRate;

    for(int gridsize = n_sm;gridsize<320*n_sm;gridsize*=2){
	    
	    char * data_gpu;
	    int *index_gpu;
	    unsigned int *time_gpu;
	    int data_size_max = gmem_total*3/4;
	    CUDA_SAFE_CALL(cudaMalloc((void**)&data_gpu, data_size_max));
	    CUDA_SAFE_CALL(cudaMemset(data_gpu, 0, data_size_max));
	    CUDA_SAFE_CALL(cudaMalloc((void**)&index_gpu, sizeof(int)*gridsize*blocksize));
	    CUDA_SAFE_CALL(cudaMemset(index_gpu, 0, sizeof(int)*gridsize*blocksize));
	    CUDA_SAFE_CALL(cudaMalloc((void**)&time_gpu, sizeof(int)*gridsize*blocksize));
	    CUDA_SAFE_CALL(cudaMemset(time_gpu, 0, sizeof(int)*gridsize*blocksize));

		int data_rowsize = data_size_unit*(gridsize*blocksize/n_hwsize);
		int round = data_size_max / data_rowsize;
		if(round <= 0)
			break;
		printf("sm=%d,half warpsize=%d,gridsize=%d,blocksize=%d\n",n_sm,n_hwsize,gridsize,blocksize);
		printf("data unit size=%d,data_rowsize=%d,data_size_max=%d,round=%d\n",data_size_unit,data_rowsize,data_size_max,round);

		for(int mode=0;mode<nmode-1;mode++){
			fillmem<<<gridsize,blocksize>>>(index_gpu,n_hwsize,data_size_unit,mode,0);

			cudaThreadSynchronize();
			double t0 = get_useconds();
			for(int i=0;i<loop;i++){
		    		bandwidth_kernel<<<gridsize,blocksize>>>(time_gpu,index_gpu,data_gpu,round,data_rowsize,0);
		    	}
			cudaThreadSynchronize();
		    	double tt = get_useconds()-t0;

		    	unsigned int blocktime = 0;
		    	CUDA_SAFE_CALL(cudaMemcpy(&blocktime,time_gpu,sizeof(int),cudaMemcpyDeviceToHost));
		    	double nstime = (double)blocktime / freq * 1000000.;

		    	unsigned long long totalsize= loop*kernelloop*elesize*blocksize*gridsize;
		    	
		    	printf("mode#%d: time %.0f ms, throughoutput=%.1f GB/s, time %.0f ms,bandwith=%.1f GB/s\n",mode,tt/1000,1.0*totalsize/tt,nstime/1000000,1000.0*kernelloop*elesize*blocksize*gridsize/nstime);
		    }

		for(int param = 0;param<256;param+=10){
			fillmem<<<gridsize,blocksize>>>(index_gpu,n_hwsize,data_size_unit,nmode-1,param);
			cudaThreadSynchronize();
			double t0 = get_useconds();
			for(int i=0;i<loop;i++){
		    		bandwidth_kernel<<<gridsize,blocksize>>>(time_gpu,index_gpu,data_gpu,round,data_rowsize,0);
		    	}
			cudaThreadSynchronize();
		    	double tt = get_useconds()-t0;

		    	unsigned int blocktime = 0;
		    	CUDA_SAFE_CALL(cudaMemcpy(&blocktime,time_gpu,sizeof(int),cudaMemcpyDeviceToHost));
		    	double nstime = (double)blocktime / freq * 1000000.;

		    	unsigned long long totalsize= loop*kernelloop*elesize*blocksize*gridsize;
		    	
		    	printf("offset=%d,gridsize=%d: time %.0f ms, throughoutput=%.1f GB/s, time %.0f ms,bandwith=%.1f GB/s\n",param,gridsize,tt/1000,1.0*totalsize/tt,nstime/1000000,1000.0*kernelloop*elesize*blocksize*gridsize/nstime);

		}
		   
	    
	     CUDA_SAFE_CALL(cudaFree(index_gpu));
	   CUDA_SAFE_CALL(cudaFree(data_gpu));
	   CUDA_SAFE_CALL(cudaFree(time_gpu));

	  }
    return 0;
}

