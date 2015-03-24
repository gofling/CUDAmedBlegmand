
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>


void fillA(int T[40][70]) {

	for (int i = 0; i < 40; i++) {
		for (int j = 0; j < 70; j++) {
			T[i][j] = rand() % 10;
		}
	}

}

void fillB(int U[70][60]) {

	for (int i = 0; i < 70; i++) {
		for (int j = 0; j < 60; j++) {
			U[i][j] = rand() % 10;
		}
	}

}


int main() {
	int A[40][70], B[70][60], C[40][60];
	int *dev_A[40][70], *dev_B[70][60], *dev_C[40][60];

	//Fill matrices
	fillA(A);
	fillB(B);

	//allocate memory on the device
	size_t sizeA = 40 * 70 * sizeof(int);
	cudaMalloc((void **)&dev_A, sizeA);
	size_t sizeB = 70 * 60 * sizeof(int);
	cudaMalloc((void **)&dev_B, sizeB);
	size_t sizeC = 40 * 60 * sizeof(int);
	cudaMalloc((void **)&dev_C, sizeC);

	//Move A and B to device memory
	//cudaMemcpy(dev_A, &A, sizeA, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_B, &B, sizeB, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t status;
	cublasOperation_t transa, transb;
	int m = 40, n = 60, k = 70;
	const float alpha = 1.0f;

	status = cublasSgemm(handle, transa, transb, m, n, k, &alpha, , );

		//cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
		//const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc


	return 0;
}