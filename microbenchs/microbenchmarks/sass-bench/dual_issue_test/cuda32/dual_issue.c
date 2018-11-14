//** framework for assamble kernel
//*  by llc
//*  NCIC, Institute of computing Technology
//*  Chinese Academy of Sciences
//*  2010.9.20 created
//*  log: ver 1.0
//*  app for display dual issue feature
//*  2011.1.13 modified
#include "export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define NUM_WARPS		(16)

#define CUBINFILE		"dual_issue_kernel.cubin"
#define cuFuncName		"func_kernel"
#define ARG_SIZE		8	

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
}/*__attribute__((aligned(8)))*/;

static int done_init = -1;
static CUdevice devHandler;
static CUcontext ctx;
static CUmodule cuMod;
static CUfunction cuFunc;

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

void measure_dual_issue(dual_issue_info *info) {
	initFrame();

	kernel_args args;

	//int NUM_WARPS = 1;
	// NUM_WARPS define as a macro
	int r_size = 16*NUM_WARPS;
	int *res = NULL;
	CUdeviceptr d_res;

	if( !(res=(int *)malloc(r_size*sizeof(int))) ) { error("alloc memory fail!");}
	memset(res, 0, r_size*sizeof(int));

	CU_SAFE_CALL_DECODED(cuMemAlloc(&d_res, r_size*sizeof(int)));
	CU_SAFE_CALL_DECODED(cuMemsetD32(d_res, 0, r_size));
	args.d_res = d_res;

	int *iadd_start, *iadd_end, *fadd_start, *fadd_end, *dmul_start, *dmul_end, *dfma_start, *dfma_end;

	if( !(iadd_start=(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(iadd_end  =(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(fadd_start=(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(fadd_end  =(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(dmul_start=(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(dmul_end  =(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(dfma_start=(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}
	if( !(dfma_end  =(int *)malloc(NUM_WARPS*sizeof(int))) ) { error("alloc memory fail!");}

	int bx, by, bz, gx, gy;
    int index=0;	
    for(int warp_i=1; warp_i<=NUM_WARPS; warp_i*=2) {
	
	bx = 32*warp_i;
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

/*
	for(int i=0; i<r_size/16; i++) {
		for(int j=0; j<16; j++) {
			printf("%d ", 2*res[i*16+j]); // SP clock is 2 times SM clock
		}
		printf("\n");
	}
*/

	int warps = bx/32;


	for(int i=0; i<warps; i++) iadd_start[i] = 2*res[i*16+1];
	for(int i=0; i<warps; i++) iadd_end[i]   = 2*res[i*16+9];
	for(int i=0; i<warps; i++) fadd_start[i] = 2*res[i*16+2];
	for(int i=0; i<warps; i++) fadd_end[i]   = 2*res[i*16+10];
	for(int i=0; i<warps; i++) dmul_start[i] = 2*res[i*16+3];
	for(int i=0; i<warps; i++) dmul_end[i]   = 2*res[i*16+11];
	for(int i=0; i<warps; i++) dfma_start[i] = 2*res[i*16+4];
	for(int i=0; i<warps; i++) dfma_end[i]   = 2*res[i*16+12];

	info->t_iadd[index] = max(iadd_end,warps)-min(iadd_start,warps) ;
	info->t_fadd[index] = max(fadd_end,warps)-min(fadd_start,warps) ;
	info->t_fmul[index] = max(dmul_end,warps)-min(dmul_start,warps) ;
	info->t_dfma[index] = max(dfma_end,warps)-min(dfma_start,warps) ;

	index++;
    }

	free(iadd_start);
	free(fadd_start);
	free(dmul_start);
	free(dfma_start);
	free(iadd_end);
	free(fadd_end);
	free(dmul_end);
	free(dfma_end);

	CU_SAFE_CALL_DECODED(cuMemFree(d_res));
	free(res);

	destroyFrame();

}

void disp_dual_issue(void) {
	dual_issue_info info;

	measure_dual_issue(&info);
	printf("excution time of 4 identical instrucions...\n\n");
    int warps=1;
    for(int index=0;index<5;index++) {
	printf("total warps: %d\n", warps);
	printf("iadd excution time: %d\n", info.t_iadd[index] );
	printf("fadd excution time: %d\n", info.t_fadd[index] );
	printf("dmul excution time: %d\n", info.t_fmul[index] );
	printf("dfma excution time: %d\n", info.t_dfma[index] );
	printf("\n");
	warps*=2;
    }

}


int main() {

	disp_dual_issue();

	return 0;
}

