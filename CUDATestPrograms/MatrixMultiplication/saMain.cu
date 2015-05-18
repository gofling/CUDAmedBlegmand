#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <time.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

/* Index to Rows */
#define IDX2R(i,j,ld) (((i)*(ld))+(j))
/* Index to Columns */
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/* Index to FORTAN (Columns with 1-indexing)*/
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
/* Index to Trinity (Rows with 1-indexing)*/
#define IDX2T(i,j,ld) ((((i)-1)*(ld))+((j)-1))

int main() {
	float h_A[6] = { 1, 2, 3, 4, 5, 6 };
	float h_B[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int rowsA, colsA, rowsB, colsB;
	rowsA = 2;
	colsA = 3;
	rowsB = 3;
	colsB = 4;
	float h_C[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	//float *h_C = (float*)calloc(rowsA * colsB, sizeof(float));

	float *d_A, *d_B, *d_C;
	int rowsC = rowsA, colsC = colsB;
	cudaError_t error;

	// Allocate memory on Device
	error = cudaMalloc(&d_A, rowsA * colsA * sizeof(float));
	if (error != cudaSuccess) {
		printf("Memory was not allocated for matrix A");
		return EXIT_FAILURE;
	}

	error = cudaMalloc(&d_B, rowsB * colsB * sizeof(float));
	if (error != cudaSuccess) {
		printf("Memory was not allocated for matrix B");
		return EXIT_FAILURE;
	}

	error = cudaMalloc(&d_C, rowsC * colsC * sizeof(float));
	if (error != cudaSuccess) {
		printf("Memory was not allocated for matrix C");
		return EXIT_FAILURE;
	}

	//Copy h_A and h_B to the device
	error = cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("Copying matrice h_A HtoD failed");
		return EXIT_FAILURE;
	}

	error = cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("Copying matrice h_B HtoD failed");
		return EXIT_FAILURE;
	}

	// Multiplication on the device
	int lda = rowsA, ldb = colsA, ldc = rowsA;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication (Matrices are stored column-major!)
	if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, colsB, colsA, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc) != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm failed");
	}

	// Destroy the handle
	cublasDestroy(handle);

	//Copy result back to the host
	error = cudaMemcpy(h_C, d_C, rowsC * colsC * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("Copying matrix d_C DtoH failed iteration");
		return EXIT_FAILURE;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();


	for (int i = 0; i < rowsA; ++i){
		for (int j = 0; j < colsB; ++j){
			printf("%f ", h_C[j * rowsA + i]);
		}
		printf("\n");
	}
	printf("\n");

	return 0;
}

//#include "kernel.cu"
/*
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
}*/