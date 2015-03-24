#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

__global__ void add(int *a, int *b, int *c, int *d, int *e, int *f) {
	*c = *a + *b;
	*d = *a - *b;
	*e = *a * *b;
	*f = *a / *b;
}
