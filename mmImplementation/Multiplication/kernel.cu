/*#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>*/


#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// This code assumes that your device support block size of 1024
#define MAX_RANGE 9999

#define funcCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "Failed to run stmt %d ", __LINE__);                       \
            printf( "Got CUDA error ...  %s ", cudaGetErrorString(err));    \
            return -1;                                                        \
		        }                                                                     \
	    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	__shared__ float sA[32][32];   // Tile size of 32x32 
	__shared__ float sB[32][32];

	int Row = blockDim.y*blockIdx.y + threadIdx.y;
	int Col = blockDim.x*blockIdx.x + threadIdx.x;
	float Cvalue = 0.0;
	sA[threadIdx.y][threadIdx.x] = 0.0;
	sB[threadIdx.y][threadIdx.x] = 0.0;

	for (int k = 0; k < (((numAColumns - 1) / 32) + 1); k++)
	{
		if ((Row < numARows) && (threadIdx.x + (k * 32)) < numAColumns)
		{
			sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k * 32)];
		}
		else
		{
			sA[threadIdx.y][threadIdx.x] = 0.0;
		}
		if (Col < numBColumns && (threadIdx.y + k * 32) < numBRows)
		{
			sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * 32)*numBColumns + Col];
		}
		else
		{
			sB[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads();

		for (int j = 0; j < 32; ++j)
		{
			Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
		}
	}
	if (Row < numCRows && Col < numCColumns)
	{
		C[Row*numCColumns + Col] = Cvalue;
	}
}

void matMultiplyOnHost(float * A, float * B, float * C, int numARows,
	int numAColumns, int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	for (int i = 0; i < numARows; i++)
	{
		for (int j = 0; j < numAColumns; j++)
		{
			C[i*numCColumns + j] = 0.0;
			for (int k = 0; k < numCColumns; k++)
			{
				C[i*numCColumns + j] += A[i*numAColumns + k] * B[k*numBColumns + j];
			}
		}
	}
	return;
}

int main(int argc, char ** argv) {
	float * hostA; // The A matrix
	float * hostB; // The B matrix
	float * hostC; // The output C matrix
	float * hostComputedC;
	float * deviceA;
	float * deviceB;
	float * deviceC;

	// Please adjust rows and columns according to you need.
	int numARows = 512; // number of rows in the matrix A
	int numAColumns = 512; // number of columns in the matrix A
	int numBRows = 512; // number of rows in the matrix B
	int numBColumns = 512; // number of columns in the matrix B

	int numCRows; // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)

	hostA = (float *)malloc(sizeof(float)*numARows*numAColumns);
	hostB = (float *)malloc(sizeof(float)*numBRows*numBColumns);

	for (int i = 0; i < numARows*numAColumns; i++)
	{
		hostA[i] = (rand() % MAX_RANGE) / 2.0;
	}
	for (int i = 0; i < numBRows*numBColumns; i++)
	{
		hostB[i] = (rand() % MAX_RANGE) / 2.0;
	}

	// Setting numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;

	hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
	hostComputedC = (float *)malloc(sizeof(float)*numCRows*numCColumns);

	// Allocating GPU memory
	funcCheck(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
	funcCheck(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns));
	funcCheck(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));

	// Copy memory to the GPU 
	funcCheck(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
	funcCheck(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice));

	// Initialize the grid and block dimensions 
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((numCColumns / 32) + 1, (numCRows / 32) + 1, 1);

	//@@ Launch the GPU Kernel here
	matrixMultiplyShared << <dimGrid, dimBlock >> >(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

	cudaError_t err1 = cudaPeekAtLastError();
	cudaDeviceSynchronize();
	printf("Got CUDA error ... %s \n", cudaGetErrorString(err1));

	// Copy the results in GPU memory back to the CPU    
	funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

	matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

	for (int i = 0; i < numCColumns*numCRows; i++)
	{
		if (hostComputedC[i] != hostC[i])
		{
			printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
			break;
		}
	}
	// Free the GPU memory
	funcCheck(cudaFree(deviceA));
	funcCheck(cudaFree(deviceB));
	funcCheck(cudaFree(deviceC));

	free(hostA);
	free(hostB);
	free(hostC);
	free(hostComputedC);

	return 0;
}


/*
#define MATRIX_MEM_SIZE(rows, cols) rows * cols * sizeof(float)
// Block must always be multiple of 32 (gpu issue instr for 32 threads)
// many threads are better for shared memory because threads share memory
// cc 3.0 enabled gpus can have 2048 threads active
#define THREADS_PER_BLOCK 1024
#define BLOCK_SIZE 16

float *matmul(float *A, int rowsA, int colsA, float *B, int rowsB, int colsB);
__global__ void gpu_matmul(float *d_A, int rowsA, int colsA, float *d_B, int rowsB, int colsB, float *d_C);
__device__ float get_element(float *A, int row, int col);
__device__ float set_element(float *A, int row, int col, float value);
__device__ float *get_sub_matrix(float *A, int row, int col);
*/
/*
int main() {
	float A[6] = {1, 2, 3, 4, 5, 6};
	float B[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; 
	int rowsA, colsA, rowsB, colsB;
	rowsA = 2;
	colsA = 3;
	rowsB = 3;
	colsB = 4;
	float *C = (float*)calloc(rowsA * colsB, sizeof(float));

	Matrix Am, Bm, Cm;
	Am.height = rowsA;
	Am.width = colsA;
	Am.elements = A;
	Am.stride = sizeof(float);
	Bm.height = rowsB;
	Bm.width = colsB;
	Bm.elements = B;
	Bm.stride = sizeof(float);
	Cm.height = rowsA;
	Cm.width = colsB;
	Cm.elements = C;
	Cm.stride = sizeof(float);

	MatMul(Am, Bm, Cm);

	//C = matmul(A, rowsA, colsA, B, rowsB, colsB);

}*/
/*
float *matmul(const float *h_A, int rowsA, int colsA, const float *h_B, int rowsB, int colsB) {
	// TODO: Make error checks for copy etc.
	int rowsC, colsC;
	rowsC = rowsA;
	colsC = colsB;
	float *d_A, *d_B, *d_C;
	float *h_C = (float*)calloc(rowsC * colsC, sizeof(float));

	// Allocate host memory for A, B and C
	cudaMalloc(&d_A, MATRIX_MEM_SIZE(rowsA, colsA));
	cudaMalloc(&d_B, MATRIX_MEM_SIZE(rowsB, colsB));
	cudaMalloc(&d_C, MATRIX_MEM_SIZE(rowsC, colsC));

	// Copy matrices to device
	cudaMemcpy(d_A, h_A, MATRIX_MEM_SIZE(rowsA, colsA), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, MATRIX_MEM_SIZE(rowsB, colsB), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(colsB / dimBlock.x, rowsA / dimBlock.y);

	gpu_matmul <<<dimGrid, dimBlock>>>(d_A, rowsA, colsA, d_B, rowsB, colsB, d_C);

	// Copy result to host
	cudaMemcpy(h_C, d_C, MATRIX_MEM_SIZE(rowsC, colsC), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return h_C;
}

__global__ void gpu_matmul(float *d_A, int rowsA, int colsA, float *d_B, int rowsB, int colsB, float *d_C) {

	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	float *Csub = get_sub_matrix(d_C, blockRow, blockCol);

	float Cvalue = 0;

	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	int m, e;

	for (m = 0; m < (colsA / BLOCK_SIZE); ++m) {

		float *Asub = get_sub_matrix(d_A, blockRow, m);

		float *Bsub = get_sub_matrix(d_B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = get_element(Asub, row, col);
		Bs[row][col] = get_element(Bsub, row, col);

		for (int e = 0; e < BLOCK_SIZE; ++e) {
			Cvalue += As[row][e] * Bs[e][col];

		__syncthreads();
		}

		set_element(Csub, row, col, Cvalue);
	}
}

__device__ float get_element(float *A, int row, int col) {
	return A[row * sizeof(float) + col];
}

__device__ float set_element(float *A, int row, int col, float value) {
	A[row * sizeof(float) + col] = value;
}

__device__ float *get_sub_matrix(float *A, int row, int col) {
	float *Asub;
	int AsubRows, AsubCols;
	AsubRows = BLOCK_SIZE;
	AsubCols = BLOCK_SIZE;
	Asub = &A[sizeof(float) * BLOCK_SIZE * row + BLOCK_SIZE * col];

	return Asub;
}

*/





/*






// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

void MatMul(const Matrix A, const Matrix B, Matrix C);
void print_matrix(Matrix *A, int nr_rows_A, int nr_cols_A);
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
	float value)
{
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
		+ BLOCK_SIZE * col];
	return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

int main() {
	float A[6] = { 1, 2, 3, 4, 5, 6 };
	float B[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int rowsA, colsA, rowsB, colsB;
	rowsA = 2;
	colsA = 3;
	rowsB = 3;
	colsB = 4;
	float *C = (float*)calloc(rowsA * colsB, sizeof(float));

	Matrix Am, Bm, Cm;
	Am.height = rowsA;
	Am.width = colsA;
	Am.elements = A;
	Am.stride = sizeof(float);
	Bm.height = rowsB;
	Bm.width = colsB;
	Bm.elements = B;
	Bm.stride = sizeof(float);
	Cm.height = rowsA;
	Cm.width = colsB;
	Cm.elements = C;
	Cm.stride = sizeof(float);

	MatMul(Am, Bm, Cm);

	for (int i = 0; i < Cm.height; ++i){
		for (int j = 0; j < Cm.width; ++j){
			printf("%f ", Cm.elements[j * Cm.height + i]);
		}
		printf("\n");
	}
	printf("\n");
	//print_matrix(Cm, Cm.height, Cm.width);
	//C = matmul(A, rowsA, colsA, B, rowsB, colsB);

}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;

	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}

void print_matrix(Matrix A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			printf("%f ", A.elements[j * nr_rows_A + i]);
		}
		printf("\n");
	}
	printf("\n");
}*/