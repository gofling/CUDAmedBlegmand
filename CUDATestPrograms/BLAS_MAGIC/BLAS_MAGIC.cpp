#include "stdafx.h"
#include "lapacke.h"
#include <stdio.h>
#include "cblas.h"
#include <stdlib.h>

#define MATRIX_IDX(n, i, j) j*n + i
#define MATRIX_ELEMENT(A, m, n, i, j) A[ MATRIX_IDX(m, i, j) ]

void init_matrix(double* A, int m, int n)
{
	double element = 1.0;
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{
			MATRIX_ELEMENT(A, m, n, i, j) = element;
			element *= 0.9;
		}
	}
}

void print_matrix(const double* A, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%8.4f", MATRIX_ELEMENT(A, m, n, i, j));
		}
		printf("\n");
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
		const int m = 3;
		const int n = 4;
		const int k = 5;

		double A[m * k];
		double B[k * n];
		double C[m * n];

		init_matrix(A, m, k);
		init_matrix(B, k, n);

		printf("Matrix A (%d x %d) is:\n", m, k);
		print_matrix(A, m, k);

		printf("\nMatrix B (%d x %d) is:\n", k, n);
		print_matrix(B, k, n);

		LAPACKE_sgemm('N', 'N', m, n, k, 1.0, A, m, B, k, 0.0, C, m);

		LAPACKE_dgesv

		printf("\nMatrix C (%d x %d) = AB is:\n", m, n);
		print_matrix(C, m, n);

		return 0;
}

