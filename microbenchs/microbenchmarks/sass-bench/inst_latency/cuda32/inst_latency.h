#ifndef _INST_LATENCY_H_
#define _INST_LATENCY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

//void error( char *message ) ;
//void msg(const char * m) ;

//static int initFrame(void) ;

void measure_inst_latency(inst_latency_info *info) ;

void disp_inst_latency(void) ;

//static int destroyFrame(void) ;

#endif
