//** framework for assamble kernel
//*  by llc
//*  NCIC, Institute of computing Technology
//*  Chinese Academy of Sciences
//*  2010.9.20 created
//*  log: ver 1.0
//*  app for measuring instruction latency
//*  2011.1.13 modified

#include "export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define NUM_WARPS		(1)

#define CUBINFILE		"inst_latency_kernel.cubin"
#define cuFuncName		"func_kernel"
#define ARG_SIZE		16	

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }

#define CU_SAFE_CALL_DECODED( call ) {                                     \
    CUresult err = call;                                                     \
    if( CUDA_SUCCESS != err) {                                               \
        fprintf(stderr, "Cuda driver error in file '%s' in line %i: %s.\n",   \
                 __FILE__, __LINE__, cudaGetErrorString((cudaError_t)err) );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }


typedef struct kernel_args {
	CUdeviceptr d_res;
	CUdeviceptr d_src;
}/*__attribute__((aligned(8)))*/;

static int done_init = -1;
static CUdevice devHandler;
static CUcontext ctx;
static CUmodule cuMod;
static CUfunction cuFunc;

//static const unsigned char cubin_bits[] = {
//#include "my_kernel_sass.cubin_bits.h"
//};

void error( char *message ) {
	fprintf( stderr, "ERROR: %s\n", message );
	fflush(stderr);
	exit (-1);
}

void msg(const char * m) {
	printf(m); printf("\n"); fflush(stdout);
}

int max(int *data, int size) {
	if(size<=0) return INT_MAX;
	int ret = data[0];
	if(size==1) return ret;

	for(int i=1; i<size; i++) 
		if(data[i]>ret) ret = data[i];

	return ret;
}

int min(int *data, int size) {
	if(size<=0) return INT_MAX;
	int ret = data[0];
	if(size==1) return ret;

	for(int i=1; i<size; i++) 
		if(data[i]<ret) ret = data[i];

	return ret;
}

static int initFrame(void) {

	//CU_SAFE_CALL_DECODED(cuInit(0));
	//int deviceCount = 0;
	//CU_SAFE_CALL_DECODED(cuDeviceGetCount(&deviceCount));
	//if(deviceCount==0) { error("No device found!"); }

	// this is the standard trick for making sure the runtime has been initialized (so we can
	// borrow its context)
	cudaFree(0);
	
	//CU_SAFE_CALL_DECODED(cuDeviceGet(&devHandler,0));
	//CU_SAFE_CALL_DECODED(cuCtxCreate(&ctx,0,devHandler));

	CU_SAFE_CALL_DECODED(cuCtxAttach(&ctx, 0));

	//CU_SAFE_CALL_DECODED(cuModuleLoadData(&cuMod, cubin_bits));
	CU_SAFE_CALL_DECODED(cuModuleLoad(&cuMod, CUBINFILE));
	CU_SAFE_CALL_DECODED(cuModuleGetFunction(&cuFunc, cuMod, cuFuncName));

	done_init = 0;
	return 0;
}

static int destroyFrame(void) {
	//CU_SAFE_CALL_DECODED(cuCtxDestroy(ctx));
	CU_SAFE_CALL_DECODED(cuCtxDetach(ctx));
	return 0;
}

void measure_inst_latency(inst_latency_info *info) {

	initFrame();

	kernel_args args;

	//int NUM_WARPS = 1;
	// NUM_WARPS define as a macro
	int r_size = 16*NUM_WARPS;
	int src_size = 32*NUM_WARPS*4*2;	// use 128bit load, and alloc duplicate space 
	int *src= NULL;
	CUdeviceptr d_src;
	int *res = NULL;
	CUdeviceptr d_res;

	if( !(src=(int *)malloc(src_size*sizeof(int))) ) { error("alloc memory fail!");}
	memset(src, 0, src_size*sizeof(int));

	if( !(res=(int *)malloc(r_size*sizeof(int))) ) { error("alloc memory fail!");}
	memset(res, 0, r_size*sizeof(int));

	CU_SAFE_CALL_DECODED(cuMemAlloc(&d_src, src_size*sizeof(int)));
	CU_SAFE_CALL_DECODED(cuMemsetD32(d_src, 0, src_size));
	args.d_src = d_src;

	CU_SAFE_CALL_DECODED(cuMemAlloc(&d_res, r_size*sizeof(int)));
	CU_SAFE_CALL_DECODED(cuMemsetD32(d_res, 0, r_size));
	args.d_res = d_res;

	int bx, by, bz, gx, gy;

    //for(int warp_i=1; warp_i<=NUM_WARPS; warp_i*=2) {

	//bx = 32*warp_i;
	bx = 32;
	by = 1;
	bz = 1;
	gx = 1;
	gy = 1;
	
	CU_SAFE_CALL_DECODED(cuParamSetv(cuFunc, 0, &args, ARG_SIZE));
	//CU_SAFE_CALL_DECODED(cuParamSetv(cuFunc, 0, &dummy, sizeof(int)));
	//CU_SAFE_CALL_DECODED(cuParamSetv(cuFunc, 8, &d_res, sizeof(CUdeviceptr)));
	CU_SAFE_CALL_DECODED(cuParamSetSize(cuFunc, ARG_SIZE));
	CU_SAFE_CALL_DECODED(cuFuncSetBlockShape(cuFunc, bx, by, bz));
	
	CU_SAFE_CALL_DECODED(cuLaunchGrid(cuFunc, gx, gy));
	
	CU_SAFE_CALL_DECODED(cuMemcpyDtoH(res, d_res, r_size*sizeof(int)));


	for(int i=0; i<r_size/16; i++) {
		for(int j=0; j<16; j++) {
			//printf("%d  ", 2*res[i*16+j]); 
			res[i*16+j] *=2;// SP clock is 2 times SM clock
		}
		//printf("\n");
	}

	
    //}
	int reissue_time = info->t_reissue = res[1]-res[0];
	info->t_ld_gm128_L2bypass_raw	= res[2]-2*reissue_time;
	info->t_ld_gm128_L2hit_raw	= res[3]-2*reissue_time;
	info->t_ld_gm128_war		= res[4]-2*reissue_time;
	info->t_ld_sm128_raw		= res[5]-2*reissue_time;
	info->t_ld_sm128_war		= res[6]-2*reissue_time;
	info->t_st_sm128_raw		= res[7]-res[5];
	info->t_st_sm128_war		= res[8]-2*reissue_time;
	info->t_iadd			= res[9]-2*reissue_time;
	info->t_dfma			= res[10]-2*reissue_time;
	info->t_branch			= res[11]-2*reissue_time;

	if( abs(res[13]-reissue_time)+abs(res[14]-reissue_time)+abs(res[15]-reissue_time)!=0 ) {
		fprintf(stderr,"Warning: instruction reissue time needs to be rechecked, ref value: 6.");
		fflush(stdout);fflush(stderr);
	}


	CU_SAFE_CALL_DECODED(cuMemFree(d_src));
	CU_SAFE_CALL_DECODED(cuMemFree(d_res));
	free(res);

	destroyFrame();
}

void disp_inst_latency(void) {
	inst_latency_info info;

	measure_inst_latency(&info);
	printf("instruction latency info:\n");
	printf("reissue\t\t\t\t%d\n",info.t_reissue);
	printf("iadd\t\t\t\t%d\n",info.t_iadd);
	printf("dfma\t\t\t\t%d\n",info.t_dfma);
	printf("branch\t\t\t\t%d\n",info.t_branch);
	printf("ld smem128 raw\t\t\t%d\n",info.t_ld_sm128_raw);
	printf("ld smem128 war\t\t\t%d\n",info.t_ld_sm128_war);
	printf("st smem128 raw\t\t\t%d\n",info.t_st_sm128_raw);
	printf("st smem128 war\t\t\t%d\n",info.t_st_sm128_war);
	printf("ld gmem128 raw L2 cache bypass\t%d\n",info.t_ld_gm128_L2bypass_raw);
	printf("ld gmem128 raw L2 cache hit\t%d\n",info.t_ld_gm128_L2hit_raw);
	printf("ld gmem128 war\t\t\t%d\n",info.t_ld_gm128_war);
	printf("\n");
}

int main() {

	disp_inst_latency();

	return 0;
}


