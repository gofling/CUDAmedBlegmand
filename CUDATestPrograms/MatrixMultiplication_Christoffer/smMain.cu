#include <cstdlib>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

//To run wite "nvcc mmul_1.cu -lcublas -lcurand -o mmul_1"

//Random filling og arrays on GPU
void GPU_fill_rand(float *A, int nrRowsA, int nrColsA) {

	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	curandGenerateUniform(prng, A, nrRowsA * nrColsA);
}

void GPU_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = n;
	const float alf = 0;
	const float bet = 1;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);
	
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	cublasDestroy(handle);
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			printf("%f ", A[j * nr_rows_A + i]);
		}

			printf("\n");
	}
	printf("\n");
}

int main() {

	// 3 Arrays on CPU
	int nrRowsA, nrColsA, nrRowsB, nrColsB, nrRowsC, nrColsC;

	// Square Arrays
	nrRowsA = nrColsA = nrRowsB = nrColsB = nrRowsC = nrColsC = 3;

	// Allocate memory on CPU
	float * h_A = (float*)malloc(nrRowsA * nrColsA * sizeof(float));
	float * h_B = (float*)malloc(nrRowsB * nrColsB * sizeof(float));
	float * h_C = (float*)malloc(nrRowsB * nrColsB * sizeof(float));

	// Allocate memory on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, nrRowsA * nrColsA * sizeof(float));
	cudaMalloc(&d_B, nrRowsB * nrColsB * sizeof(float));
	cudaMalloc(&d_C, nrRowsC * nrColsC * sizeof(float));


	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nrRowsA, nrColsA);
	GPU_fill_rand(d_B, nrRowsB, nrColsB);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A, d_A, nrRowsA * nrColsA * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, nrRowsB * nrColsB * sizeof(float), cudaMemcpyDeviceToHost);
	printf("A = \n");
	print_matrix(h_A, nrRowsA, nrColsA);
	printf("B = \n");
	print_matrix(h_B, nrRowsB, nrColsB);

	// Multiply A and B on GPU
	GPU_blas_mmul(d_A, d_B, d_C, nrRowsA, nrColsA, nrColsB);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C, d_C, nrRowsC * nrColsC * sizeof(float), cudaMemcpyDeviceToHost);
	printf("C = \n");
	print_matrix(h_C, nrRowsC, nrColsC);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}