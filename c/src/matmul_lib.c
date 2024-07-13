#include "../include/matmul_lib.h"
#include <stdlib.h>
#include <omp.h>

// Similar to Numpy but this function will generate a uniform distribution between -1 and 1
// However, Numpy + Python will generate a normal distribution (bell curve centered at 0) while this function will not
// If we want a normal distribution in C we would need to change this function
void init_matrix(float matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

void zero_matrix(float matrix[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0.0f;
        }
    }
}

void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_blocked(float A[N][N], float B[N][N], float C[N][N]) {
	#pragma omp parallel for collapse(2)
	for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
				for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
					for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
						for (int k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
							C[i][j] += A[i][k] * B[k][j];
						}
					}
				}
			}
		}
	}
}
