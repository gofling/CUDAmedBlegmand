#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <time.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

//Random filling og arrays on device
void GPU_fill_rand(float *A, int nrRowsA, int nrColsA) {

	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	curandGenerateUniform(prng, A, nrRowsA * nrColsA);
}

// Function that multiplies matrices on the device 
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

// Function that multiplies matrices on the host
static void cpu_blas_mmul(int n, const float *A, const float *B, float *C)
{
	const float alpha = 1.0f, beta = 0.0f;
	int i;
	int j;
	int k;

	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			float prod = 0;

			for (k = 0; k < n; ++k)
			{
				prod += A[k * n + i] * B[j * n + k];
			}

			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
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

void output_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	FILE *f = fopen("./Output.txt", "w");
	if (f == NULL) {
		printf("Yo dude! Does not load da file!");
		exit(1);
	}
	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			fprintf(f, "%f ", A[j * nr_rows_A + i]);
		}

		fprintf(f, "\n");
	}
	fprintf(f, "\n\n");

	fclose(f);
}

int main() {

	// 3 Arrays on CPU
	int nrRowsA, nrColsA, nrRowsB, nrColsB, nrRowsC, nrColsC, nrRowsD, nrColsD, sizeAsInput = 1;

	// Square Arrays
	printf("Square matrices of size: ");
	scanf("%d", &sizeAsInput);
	nrRowsA = nrColsA = nrRowsB = nrColsB = nrRowsC = nrColsC = nrRowsD = nrColsD = sizeAsInput;

	// Allocate memory on Host
	clock_t start = clock(), diff;
	float * h_A = (float*)malloc(nrRowsA * nrColsA * sizeof(float));
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("CPU: A allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	start = clock(), diff;
	float * h_B = (float*)malloc(nrRowsB * nrColsB * sizeof(float));
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("CPU: B allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	start = clock(), diff;
	float * h_C = (float*)malloc(nrRowsB * nrColsB * sizeof(float));
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("CPU: C allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	start = clock(), diff;
	float * h_D = (float*)malloc(nrRowsD * nrColsD * sizeof(float));
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("CPU: D allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	// Allocate memory on Device
	float *d_A, *d_B, *d_C;
	//printf("GPU memory allocation times\n");

	// Memory allocation for Matrix A
	start = clock(), diff;
	if (cudaMalloc(&d_A, nrRowsA * nrColsA * sizeof(float)) != cudaSuccess) {
		printf("Memory was not allocated for matrix A");
		return EXIT_FAILURE;
	}
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("A allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	// Memory allocation for Matrix B
	start = clock(), diff;
	if (cudaMalloc(&d_B, nrRowsB * nrColsB * sizeof(float)) != cudaSuccess) {
		printf("Memory was not allocated for matrix B");
		return EXIT_FAILURE;
	}
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("B allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	// Memory allocation for Matrix C
	start = clock(), diff;
	if (cudaMalloc(&d_C, nrRowsC * nrColsC * sizeof(float)) != cudaSuccess) {
		printf("Memory was not allocated for matrix C");
		return EXIT_FAILURE;
	}
	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("C allocation time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);


	// Fill the arrays A and B on GPU with random numbers
	//GPU_fill_rand(d_A, nrRowsA, nrColsA);
	//GPU_fill_rand(d_B, nrRowsB, nrColsB);

	// Optionally we can copy the data back on CPU and print the arrays
	//start = clock(), diff;
	/*if (cudaMemcpy(h_A, d_A, nrRowsA * nrColsA * sizeof(float), cudaMemcpyDeviceToHost) || cudaMemcpy(h_B, d_B, nrRowsB * nrColsB * sizeof(float), cudaMemcpyDeviceToHost) != CUBLAS_STATUS_SUCCESS){
		printf("Copying matrice A or B failed.\n");
		return EXIT_FAILURE;
	}*/
	//diff = clock() - start;
	//msec = diff * 1000 / CLOCKS_PER_SEC;
	//printf("Move random filled arrays from GPU to CPU time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	//printf("A = \n");
	//print_matrix(h_A, nrRowsA, nrColsA);
	//printf("B = \n");
	//print_matrix(h_B, nrRowsB, nrColsB);

	// Multiply A and B on device
	//start = clock(), diff;
	//gpu_blas_mmul(d_A, d_B, d_C, nrRowsA, nrColsA, nrColsB);
	//diff = clock() - start;
	//msec = diff * 1000 / CLOCKS_PER_SEC;
	//printf("GPU time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	// Copy (and print) the result on host memory
	//start = clock(), diff;
	//cudaMemcpy(h_C, d_C, nrRowsC * nrColsC * sizeof(float), cudaMemcpyDeviceToHost);
	//output_matrix(h_C, nrRowsC, nrColsC);
	//diff = clock() - start;
	////msec = diff * 1000 / CLOCKS_PER_SEC;
	//printf("Copy result amtrix from gpu to cpu time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	//printf("GPU C = \n");
	//print_matrix(h_C, nrRowsC, nrColsC);

	// Multiply A and B on the host
	//start = clock(), diff;
	//cpu_blas_mmul(nrRowsA, h_A, h_B, h_D);
	//diff = clock() - start;
	//msec = diff * 1000 / CLOCKS_PER_SEC;
	//printf("CPU time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	//printf("CPU C = \n");
	//print_matrix(h_C, nrRowsC, nrColsC);

	//print_matrix(h_D, nrRowsD, nrColsD);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	//printf("Done John");
	//int q = 0;
	//scanf("%d", q);

	return 0;
}