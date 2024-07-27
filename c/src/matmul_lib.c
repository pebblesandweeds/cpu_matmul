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

void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]) {
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

void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
    int i, j, k;
    float A_col[4][N] __attribute__((aligned(32)));
    float B_col[4*N] __attribute__((aligned(32)));

    #pragma omp parallel for private(i, j, k, A_col, B_col)
    for (j = 0; j < N; j += 4) {
        // Convert columns of B to column-major order
        for (k = 0; k < N; k++) {
            for (int jj = 0; jj < 4 && j + jj < N; jj++) {
                B_col[jj*N + k] = B[k][j + jj];
            }
        }

        for (i = 0; i < N; i += 4) {
            // Convert rows of A to column-major order
            for (int ii = 0; ii < 4 && i + ii < N; ii++) {
                for (k = 0; k < N; k++) {
                    A_col[ii][k] = A[i+ii][k];
                }
            }

            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c13 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c22 = _mm256_setzero_ps();
            __m256 c23 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            __m256 c32 = _mm256_setzero_ps();
            __m256 c33 = _mm256_setzero_ps();

            for (k = 0; k < N; k += 8) {
                __m256 a0 = _mm256_loadu_ps(&A_col[0][k]);
                __m256 a1 = _mm256_loadu_ps(&A_col[1][k]);
                __m256 a2 = _mm256_loadu_ps(&A_col[2][k]);
                __m256 a3 = _mm256_loadu_ps(&A_col[3][k]);

                __m256 b0 = _mm256_loadu_ps(&B_col[0*N + k]);
                __m256 b1 = _mm256_loadu_ps(&B_col[1*N + k]);
                __m256 b2 = _mm256_loadu_ps(&B_col[2*N + k]);
                __m256 b3 = _mm256_loadu_ps(&B_col[3*N + k]);

                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c02 = _mm256_fmadd_ps(a0, b2, c02);
                c03 = _mm256_fmadd_ps(a0, b3, c03);

                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                c12 = _mm256_fmadd_ps(a1, b2, c12);
                c13 = _mm256_fmadd_ps(a1, b3, c13);

                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                c22 = _mm256_fmadd_ps(a2, b2, c22);
                c23 = _mm256_fmadd_ps(a2, b3, c23);

                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
                c32 = _mm256_fmadd_ps(a3, b2, c32);
                c33 = _mm256_fmadd_ps(a3, b3, c33);
            }

            for (int ii = 0; ii < 4 && i + ii < N; ii++) {
                __m256 ci0, ci1, ci2, ci3;
                if (ii == 0) {
                    ci0 = c00; ci1 = c01; ci2 = c02; ci3 = c03;
                } else if (ii == 1) {
                    ci0 = c10; ci1 = c11; ci2 = c12; ci3 = c13;
                } else if (ii == 2) {
                    ci0 = c20; ci1 = c21; ci2 = c22; ci3 = c23;
                } else {
                    ci0 = c30; ci1 = c31; ci2 = c32; ci3 = c33;
                }

                for (int jj = 0; jj < 4 && j + jj < N; jj++) {
                    __m256 cij;
                    if (jj == 0) cij = ci0;
                    else if (jj == 1) cij = ci1;
                    else if (jj == 2) cij = ci2;
                    else cij = ci3;

                    __m128 sum_low = _mm256_castps256_ps128(cij);
                    __m128 sum_high = _mm256_extractf128_ps(cij, 1);
                    __m128 sum = _mm_add_ps(sum_low, sum_high);
                    sum = _mm_hadd_ps(sum, sum);
                    sum = _mm_hadd_ps(sum, sum);
                    C[i+ii][j+jj] += _mm_cvtss_f32(sum);
                }
            }
        }
    }
}
