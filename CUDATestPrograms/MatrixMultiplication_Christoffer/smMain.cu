#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <time.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

//Prototypes
void GPU_fill_rand(float *A, int nrRowsA, int nrColsA);
void CPU_fill_matrices(float* A, int nrRowsA, int nrColsA);
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A);
void fprint_MemCpy_Times(int matrixSize, int iterationnr, int msec, char *currentMatrix, char *fileName);
void fprint_sgemm_time(int matrixSize, int iterationnr, int msec, char *fileName);
void output_matrix(const float *A, int nr_rows_A, int nr_cols_A, char *fileName);

int main() {
	printf("Initializing...\n");
	int nrRowsA, nrColsA, nrRowsB, nrColsB, nrRowsC, nrColsC;
	int matrixStartSize = 500,
		matrixMaxSize = 12000,
		matrixIncrease = 500,
		sgemmIterations = 50,
		sgemmIterationsDecrease = 5;
	int matrixActualSize = matrixStartSize;
	float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
	srand(time(NULL));
	cudaError_t error;
	printf("Copying from matrix size %d to %d.\n", matrixStartSize, matrixMaxSize);
	printf("Increasing size with %d for each iteration.\n\n", matrixIncrease);

	// Calculations
	printf("Initializing complete. Starting calculations...\n");
	while (matrixActualSize <= matrixMaxSize){
		printf("Calculating with size %d: ", matrixActualSize);

		// Square Arrays
		nrRowsA = nrColsA = nrRowsB = nrColsB = nrRowsC = nrColsC = matrixActualSize;

		for (int k = 0; k < sgemmIterations; k++){
			/*if (k % 10 == 0)
			{
				printf("%d ", k);
			}*/

			// Allocate memory on Host
			h_A = (float*)malloc(nrRowsA * nrColsA * sizeof(float));
			if (h_A == NULL) { 
				printf("CPU: h_A was not allocated: %d", k); 
				return EXIT_FAILURE; 
			}

			h_B = (float*)malloc(nrRowsB * nrColsB * sizeof(float));
			if (h_B == NULL) { 
				printf("CPU: h_B was not allocated: %d", k); 
				return EXIT_FAILURE; 
			}
		
			h_C = (float*)malloc(nrRowsC * nrColsC * sizeof(float));
			if (h_C == NULL) { 
				printf("CPU: h_C was not allocated: %d", k); 
				return EXIT_FAILURE; 
			}
			
			// Allocate memory on Device
			error = cudaMalloc(&d_A, nrRowsA * nrColsA * sizeof(float));
			if (error != cudaSuccess) {
				printf("Memory was not allocated for matrix A: %d", k);
				return EXIT_FAILURE;
			}

			error = cudaMalloc(&d_B, nrRowsB * nrColsB * sizeof(float));
			if (error != cudaSuccess) {
				printf("Memory was not allocated for matrix B: %d", k);
				return EXIT_FAILURE;
			}

			error = cudaMalloc(&d_C, nrRowsC * nrColsC * sizeof(float));
			if (error != cudaSuccess) {
				printf("Memory was not allocated for matrix C: %d", k);
				return EXIT_FAILURE;
			}

			// Fill the arrays A and B on GPU with random numbers
			//GPU_fill_rand(d_A, nrRowsA, nrColsA);
			//GPU_fill_rand(d_B, nrRowsB, nrColsB);
			CPU_fill_matrices(h_A, nrRowsA, nrColsA);
			CPU_fill_matrices(h_B, nrColsB, nrColsB);

			//Copy h_A and h_B to the device
			clock_t start = clock(), diff;
			error = cudaMemcpy(d_A, h_A, nrRowsA * nrColsA * sizeof(float), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
				printf("Copying matrice h_A HtoD failed.\n: %d", k);
				return EXIT_FAILURE;
			}
			diff = clock() - start;
			int msec = diff * 1000 / CLOCKS_PER_SEC;
			fprint_MemCpy_Times(matrixActualSize, k, msec, "MemCpy:A", "./MemCpyHtoDTimes.txt");

			start = clock(), diff;
			error = cudaMemcpy(d_B, h_B, nrRowsB * nrColsB * sizeof(float), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
				printf("Copying matrice h_B HtoD failed.\n: %d", k);
				return EXIT_FAILURE;
			}
			diff = clock() - start;
			msec = diff * 1000 / CLOCKS_PER_SEC;
			fprint_MemCpy_Times(matrixActualSize, k, msec, "MemCpy:B", "./MemCpyHtoDTimes.txt");

			//Perform Sgemm on the device
			start = clock(), diff;
			gpu_blas_mmul(d_A, d_B, d_C, nrRowsA, nrColsA, nrColsB);
			diff = clock() - start;
			msec = diff * 1000 / CLOCKS_PER_SEC;
			fprint_sgemm_time(matrixActualSize, k, msec, "./SgemmGPUtimes.txt");

			//Copy result back to the host
			start = clock(), diff;
			error = cudaMemcpy(h_C, d_C, nrRowsC * nrColsC * sizeof(float), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess){
				printf("Copying matrix d_C DtoH failed iteration %d", k);
				return EXIT_FAILURE;
			}
			msec = clock() - start;
			msec = diff * 1000 / CLOCKS_PER_SEC;
			fprint_MemCpy_Times(matrixActualSize, k, msec, "MemCpy:d_C", "./MemCpyResulttoH.txt");

			//Free GPU memory
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_C);
			cudaDeviceReset();

			//Free CPU memory
			free(h_A);
			free(h_B);
			free(h_C);

		}
		printf("- Size %d done!\n", matrixActualSize);

		if (sgemmIterations > 5) {
			sgemmIterations -= sgemmIterationsDecrease;
		}

		matrixActualSize += matrixIncrease;
	}

	printf("Done John\n");
	printf("Press any key to exit...");
	getchar();

	return 0;
}

//Random fill matrices on device
void GPU_fill_rand(float *A, int nrRowsA, int nrColsA) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, A, nrRowsA * nrColsA);
}

//Random fill matrices on host
void CPU_fill_matrices(float* A, int nrRowsA, int nrColsA) {

	for (int r = 0; r < nrRowsA; r++) {
		for (int c = 0; c < nrColsA; c++){
			A[r * nrRowsA + c] = static_cast<float>(rand() % 20);
		}
	}
}

//Function that multiplies matrices on the device 
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

//Print a given host matrix
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			printf("%f ", A[j * nr_rows_A + i]);
		}
		printf("\n");
	}
	printf("\n");
}

// Print times to a .txt
void fprint_MemCpy_Times(int matrixSize, int iterationnr, int msec, char *currentMatrix, char *fileName) {
	FILE *f = fopen(fileName, "a");

	if (f == NULL) {
		printf("an error occured when opening GPUMemCopyTimes.txt\n");
		printf("Press any key to exit...");
		getchar();
		exit(1);
	}
	fprintf(f, "iteration-%d-", iterationnr);
	fprintf(f, "matrixSize-%d-", matrixSize);
	fprintf(f, "matrixName-%s-time-%d%d\n", currentMatrix, msec / 1000, msec % 1000);

	fclose(f);
}

// Print times to a .txt
void fprint_sgemm_time(int matrixSize, int iterationnr, int msec, char *fileName) {
	FILE *f = fopen(fileName, "a");

	if (f == NULL) {
		printf("an error occured when opening GPUMemCopyTimes.txt\n");
		printf("Press any key to exit...");
		getchar();
		exit(1);
	}
	fprintf(f, "iteration-%d-", iterationnr);
	fprintf(f, "matrixSize-%d-", matrixSize);
	fprintf(f, "time-%d%d\n", msec / 1000, msec % 1000);

	fclose(f);
}

// Print a given matrix's entries to a file
void output_matrix(const float *A, int nr_rows_A, int nr_cols_A, char *fileName) {
	FILE *f = fopen(fileName, "a");

	if (f == NULL) {
		printf("an error occured when opening a file\n");
		printf("Press any key to exit...");
		getchar();
		exit(1);
	}

	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			fprintf(f, "%f, ", A[j * nr_rows_A + i]);
		}
		fprintf(f, "\n");
	}
	fprintf(f, "\n\n");

	fclose(f);
}

