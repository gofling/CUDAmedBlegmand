#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

//#include "kernel.cu"

__global__ void add(int *a, int *b, int *c, int *d, int *e, int *f) {
	*c = *a + *b;
	*d = *a - *b;
	*e = *a * *b;
	*f = *a / *b;
}

int main(){
	
	int a, b, c, d, e, f;
	int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e, *dev_f;
	int size = sizeof(int);

	cudaMalloc((void **)&dev_a, size);
	cudaMalloc((void **)&dev_b, size);
	cudaMalloc((void **)&dev_c, size);
	cudaMalloc((void **)&dev_d, size);
	cudaMalloc((void **)&dev_e, size);
	cudaMalloc((void **)&dev_f, size);

	a = 7;
	b = 2;

	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

	add << <1, 1 >> >(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f);

	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&d, dev_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&e, dev_e, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&f, dev_f, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	cudaFree(dev_e);
	cudaFree(dev_f);

	printf("Addition: %d\nSubtraction: %d\nMultiplication: %d\nDividation: %d", c, d, e, f);

	return 0;
}