//** framework for assamble kernel
//*  by llc
//*  NCIC, Institute of computing Technology
//*  Chinese Academy of Sciences
//*  2010.9.20 created
//*  log: ver 1.0
//*  app for measuring smem load latency
//*  2011.1.20 modified

#include "frame.h"

#define CUBINFILE		"my_kernel_sass.cubin"
#define cuFuncName		"func_kernel"
#define NWARPS			(1)
#define ARG_SIZE		(8)	

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

void excute(void) {
	my_assamble();
}

static int destroyFrame(void) {
	//CU_SAFE_CALL_DECODED(cuCtxDestroy(ctx));
	CU_SAFE_CALL_DECODED(cuCtxDetach(ctx));
	return 0;
}

void my_assamble() {
	kernel_args args;

	int r_size = NWARPS*16;
	int *res = NULL;
	CUdeviceptr d_res;

	if( !(res=(int *)malloc(r_size*sizeof(int))) ) { error("alloc memory fail!");}
	memset(res, 0, r_size*sizeof(int));

	CU_SAFE_CALL_DECODED(cuMemAlloc(&d_res, r_size*sizeof(int)));
	CU_SAFE_CALL_DECODED(cuMemsetD32(d_res, 0, r_size));
	args.d_res = d_res;

	int bx, by, bz, gx, gy;
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

/*
	for(int i=0; i<r_size/16; i++) {
		for(int j=0; j<16; j++) {
			printf("%d\t", 2*res[i*16+j]); // SP clock is 2 times SM clock
		}
		printf("\n");
	}
*/
	int t_reissue = 2*res[0];
	if( !(t_reissue==2*res[6] && t_reissue==2*res[7]) )
		fprintf(stderr,"Warning: instruction reissue latency needs to be checked!\n");

	printf("\nreissue latency:\t%d\n",t_reissue);
	printf("iadd latency:\t\t%d\n",2*res[1]-2*t_reissue);
	printf("ld.sm.32 raw latency:\t%d\n",2*res[2]-2*t_reissue);
	printf("ld.sm.64 raw latency:\t%d\n",2*res[3]-2*t_reissue);
	printf("ld.sm.128 raw latency:\t%d\n",2*res[4]-2*t_reissue);

	CU_SAFE_CALL_DECODED(cuMemFree(d_res));
	free(res);
}

int main() {
	initFrame();

	excute();

	destroyFrame();

	return 0;
}
