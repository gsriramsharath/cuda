#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void measure_empty();
void measure_clock();
void measure_pipeline();
//void measure_regfile();


int main()
{

	measure_empty();
	measure_clock();
	measure_pipeline();
//	measure_regfile();

	return 0;
}


