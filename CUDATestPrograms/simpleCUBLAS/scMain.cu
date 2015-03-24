
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



	/*Matrix d_A;
d_A.width = d_A.stride = A.width;
d_A.height = A.height;
size_t size = A.width * A.height * sizeof(float);
cudaError_t err = cudaMalloc(&d_A.elements, size);
printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);*/

	return 0;
}