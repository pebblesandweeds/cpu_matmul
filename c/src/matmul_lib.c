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
    #pragma omp parallel
    {
        float A_col[8][N] __attribute__((aligned(32)));
        float B_col[8][N] __attribute__((aligned(32)));

        #pragma omp for
        for (int j = 0; j < N; j += 8) {
            // Convert columns of B to column-major order
            for (int k = 0; k < N; k++) {
                for (int jj = 0; jj < 8 && j + jj < N; jj++) {
                    B_col[jj][k] = B[k][j + jj];
                }
            }

            for (int i = 0; i < N; i += 8) {
                // Convert rows of A to column-major order
                for (int ii = 0; ii < 8 && i + ii < N; ii++) {
                    float *A_col_ptr = A_col[ii];
                    for (int k = 0; k < N; k++) {
                        *A_col_ptr++ = A[i+ii][k];
                    }
                }

                __m256 c[8][8];
                for (int ii = 0; ii < 8; ii++) {
                    for (int jj = 0; jj < 8; jj++) {
                        c[ii][jj] = _mm256_setzero_ps();
                    }
                }

                for (int k = 0; k < N; k += 32) {
                    // Prefetch next A_col and B_col data
                    if (k + 128 < N) {
                        for (int ii = 0; ii < 8; ii++) {
                            _mm_prefetch((char*)&A_col[ii][k + 128], _MM_HINT_T1);
                            _mm_prefetch((char*)&B_col[ii][k + 128], _MM_HINT_T1);
                        }
                    }

                    __m256 a[8][4], b[8][4];
                    for (int ii = 0; ii < 8; ii++) {
                        float *A_ptr = &A_col[ii][k];
                        float *B_ptr = &B_col[ii][k];
                        for (int kk = 0; kk < 4; kk++) {
                            a[ii][kk] = _mm256_load_ps(A_ptr);
                            b[ii][kk] = _mm256_load_ps(B_ptr);
                            A_ptr += 8;
                            B_ptr += 8;
                        }
                    }

                    for (int ii = 0; ii < 8; ii++) {
                        for (int jj = 0; jj < 8; jj++) {
                            c[ii][jj] = _mm256_fmadd_ps(a[ii][0], b[jj][0], c[ii][jj]);
                            c[ii][jj] = _mm256_fmadd_ps(a[ii][1], b[jj][1], c[ii][jj]);
                            c[ii][jj] = _mm256_fmadd_ps(a[ii][2], b[jj][2], c[ii][jj]);
                            c[ii][jj] = _mm256_fmadd_ps(a[ii][3], b[jj][3], c[ii][jj]);
                        }
                    }
                }

                // Reduce and store results back to C
                for (int ii = 0; ii < 8 && i + ii < N; ii++) {
                    for (int jj = 0; jj < 8 && j + jj < N; jj++) {
                        __m256 cij = c[ii][jj];
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
}
