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

int cublas_set_up(const float *h_A, int rowsA, int colsA, const float *h_B, int rowsB, int colsB, float *h_C);
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);

int main() {
	float h_A[6] = { 1, 2, 3, 4, 5, 6 };
	float h_B[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int rowsA, colsA, rowsB, colsB;
	rowsA = 2;
	colsA = 3;
	rowsB = 3;
	colsB = 4;
	float *h_C = (float*)calloc(rowsA * colsB, sizeof(float));

	cublas_set_up(h_A, rowsA, colsA, h_B, rowsB, colsB, h_C);

	for (int i = 0; i < rowsA; ++i){
		for (int j = 0; j < colsA; ++j){
			printf("%f ", h_A[j * rowsA + i]);
		}
		printf("\n");
	}
	printf("\n");


	for (int i = 0; i < rowsA; ++i){
		for (int j = 0; j < colsB; ++j){
			printf("%f ", h_C[j * rowsA + i]);
		}
		printf("\n");
	}
	printf("\n");

	return 0;
}

// TODO: exit ved fejl
int cublas_set_up(const float *h_A, int rowsA, int colsA, const float *h_B, int rowsB, int colsB, float *h_C) {
	float *d_A, *d_B, *d_C;
	int rowsC = rowsA, colsC = colsB;
	cudaError_t error;

	// Rearrange h_A and h_B to be column-major
	// TODO: skal countere initialiseres udenfor for loops??


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
	gpu_blas_mmul(d_A, d_B, d_C, rowsA, colsA, colsB);

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

	return 0;
}

// m = rowsArowsC, k = colsArowsB, n = colsBcolsC
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) != CUBLAS_STATUS_SUCCESS){
		printf("cublasSgemm failed");
	}

	// Destroy the handle
	cublasDestroy(handle);
}