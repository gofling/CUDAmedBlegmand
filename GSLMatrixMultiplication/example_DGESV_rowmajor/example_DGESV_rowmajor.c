/*---------- Includes ----------*/
#include <stdio.h>
#include <C:\cygwin64\usr\include\gsl\gsl_cblas.h>
#include <time.h>
#include <stdlib.h>
#include <windows.h>

/*---------- Prototypes ----------*/
void cpu_rand(float* A, int nrRows, int nrColumns);
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A);
void wait_exit();
void fprint_sgemm_time(int matrixSize, int iterationnr, int msec, char *fileName);

/*---------- Main ----------*/
int main(void)
{
	/*---------- Set the Seed to Feed srand() to NULL ----------*/
	srand(time(NULL));
	/*---------- Square Matrix Size ----------*/
	int nrRows, nrCols;
	float *MatrixA, *MatrixB, *MatrixC;
	int matrixStartSize = 500, matrixEndSize = 6000;
	int matrixActualSize = matrixStartSize;
	int iterationsFor = 50;
	
	while (matrixActualSize <= matrixEndSize){

		nrRows = nrCols = matrixActualSize;

		printf("\nMatrix size: %d\n", matrixActualSize);
		for (int i = 0; i < iterationsFor; i++)
		{

			/*---------- Matrix Allocation ----------*/
			MatrixA = (float*)malloc(nrRows*nrCols*sizeof(float));
			if (MatrixA == NULL) {
				printf("MatrixA not allocated iteration %d", i);
				return EXIT_FAILURE;
			}

			MatrixB = (float*)malloc(nrRows*nrCols*sizeof(float));
			if (MatrixB == NULL) {
				printf("MatrixB not allocated iteration %d", i);
				return EXIT_FAILURE;
			}

			MatrixC = (float*)malloc(nrRows*nrCols*sizeof(float));
			if (MatrixC == NULL) {
				printf("MatrixC not allocated iteration %d", i);
				return EXIT_FAILURE;
			}

			/*---------- Fill Matrix A and B With Random Numbers ----------*/
			cpu_rand(MatrixA, nrRows, nrCols);
			cpu_rand(MatrixB, nrRows, nrCols);

			/*---------- Compute MatrixC = MatrixA * MatrixB, and Time the Computation ----------*/
			clock_t start = clock(), diff;
			cblas_sgemm(CblasRowMajor,
				CblasNoTrans, CblasNoTrans, nrRows, nrCols, nrRows,
				1.0, MatrixA, nrRows, MatrixB, nrCols, 0.0, MatrixC, nrRows);
			diff = clock() - start;
			int msec = diff * 1000 / CLOCKS_PER_SEC;
			fprint_sgemm_time(matrixActualSize, i, msec, "./CPU_Times.txt");

			/*---------- Print MatrixA, MatrixB, Matrix C and Time of Computation ----------*/
			printf("ERROR: DELETED '%d' FILES FROM C:/Users/Rune/Documents/GitHub/CUDAmedBlegmand\n", i + 1);
			free(MatrixA);
			free(MatrixB);
			free(MatrixC);
		}
		/*---------- Set Matrix Size For the Next Loop to +500 ----------*/
		matrixActualSize += 500;
		/*---------- Check to See If Iterations For The Next Loop is Under 5 and Set It to be 5 If It is ----------*/
		if (iterationsFor > 5) {
			iterationsFor -= 5;
		}
		
	}
	/*---------- Wait On Keypress For Exit and Formal Ending of Main ----------*/
	wait_exit();
	return 0;
}

/*---------- Function for Random Filling of Matrix ----------*/
void cpu_rand(float* A, int nrRows, int nrColumns) {

	for (int i = 0; i < nrRows; i++) {
		for (int j = 0; j < nrColumns; j++) {
			A[i * nrRows + j] = (float)(rand() % 20);
		}
	}
}

/*---------- Function for Printing of Matrix ----------*/
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i){
		for (int j = 0; j < nr_cols_A; ++j){
			printf("%f ", A[j * nr_rows_A + i]);
		}

		printf("\n");
	}
	printf("\n");
}

/*---------- Function for Stopping exiting of Console Application ----------*/
void wait_exit() {
	
	int kage;
	scanf("%d", &kage);
}
/*---------- Funtion for Writing the Timing to a File ----------*/
void fprint_sgemm_time(int matrixSize, int iterationnr, int msec, char *fileName) {
	FILE *f = fopen(fileName, "a");

	if (f == NULL) {
		printf("an error occured when opening %s\n", fileName);
		printf("Press any key to exit...");
		getchar();
		exit(1);
	}
	fprintf(f, "iteration-%d-", iterationnr);
	fprintf(f, "matrixSize-%d-", matrixSize);
	fprintf(f, "time-%d%d\n", msec / 1000, msec % 1000);

	fclose(f);
}