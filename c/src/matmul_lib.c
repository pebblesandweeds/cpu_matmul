#include "../include/matmul_lib.h"
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

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
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_blocked(float A[N][N], float B[N][N], float C[N][N]) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii += TILE_SIZE) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += TILE_SIZE) {
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk += UNROLL_FACTOR) {
                            for (int iii = ii; iii < ii + TILE_SIZE && iii < i + BLOCK_SIZE && iii < N; iii++) {
                                for (int jjj = jj; jjj < jj + TILE_SIZE && jjj < j + BLOCK_SIZE && jjj < N; jjj++) {
                                    float c_temp = C[iii][jjj];
                                    for (int kkk = kk; kkk < kk + UNROLL_FACTOR && kkk < k + BLOCK_SIZE && kkk < N; kkk++) {
                                        c_temp += A[iii][kkk] * B[kkk][jjj];
                                    }
                                    C[iii][jjj] = c_temp;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void matmul_loop_order(float A[N][N], float B[N][N], float C[N][N]) {
	int i, j, k;
    __m256 sum0, sum1, sum2, sum3;
    __m256 Aik;
    float *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
	#pragma omp parallel for collapse(2) private(i, j, k, sum0, sum1, sum2, sum3, Aik, bp0_pntr, bp1_pntr, bp2_pntr, bp3_pntr)
	for (j = 0; j < N; j += 4) {
		for (i = 0; i < N; i++) {
			sum0 = _mm256_setzero_ps();
            sum1 = _mm256_setzero_ps();
            sum2 = _mm256_setzero_ps();
            sum3 = _mm256_setzero_ps();
			bp0_pntr = &B[0][j];
            bp1_pntr = &B[0][j + 1];
            bp2_pntr = &B[0][j + 2];
            bp3_pntr = &B[0][j + 3];
			for (k = 0; k < N; k++) {
				Aik = _mm256_set1_ps(A[i][k]);
				sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(Aik, _mm256_set1_ps(*bp0_pntr)));
                sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(Aik, _mm256_set1_ps(*bp1_pntr)));
                sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(Aik, _mm256_set1_ps(*bp2_pntr)));
                sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(Aik, _mm256_set1_ps(*bp3_pntr)));
				bp0_pntr += N;
                bp1_pntr += N;
                bp2_pntr += N;
                bp3_pntr += N;
			}
			float sums[8];
            _mm256_storeu_ps(sums, sum0);
            C[i][j] += sums[0];
            _mm256_storeu_ps(sums, sum1);
            C[i][j + 1] += sums[0];
            _mm256_storeu_ps(sums, sum2);
            C[i][j + 2] += sums[0];
            _mm256_storeu_ps(sums, sum3);
            C[i][j + 3] += sums[0];
		}
	}
}
