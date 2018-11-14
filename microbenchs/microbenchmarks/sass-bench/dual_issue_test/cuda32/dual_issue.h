#ifndef _MY_FRAME_H_
#define _MY_FRAME_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

void error( char *message ) ;
void msg(const char * m) ;

static int initFrame(void) ;

void excute(void) ;

static int destroyFrame(void) ;

void my_assamble() ;

#endif
