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

float sum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(lo);
}

void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
    int i, j, k;
    __m256 sum0, sum1, sum2, sum3;
    __m256 Aik0, Aik1, Aik2, Aik3;
    float *ap_pntr, *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
    float A_col[N], B_col[4*N];

    #pragma omp parallel for private(i, j, k, sum0, sum1, sum2, sum3, Aik0, Aik1, Aik2, Aik3, ap_pntr, bp0_pntr, bp1_pntr, bp2_pntr, bp3_pntr, A_col, B_col)
    for (j = 0; j < N; j += 4) {
        // Convert columns of B to column-major order
        for (k = 0; k < N; k++) {
            B_col[k] = B[k][j];
            B_col[N + k] = B[k][j + 1];
            B_col[2 * N + k] = B[k][j + 2];
            B_col[3 * N + k] = B[k][j + 3];
        }

        for (i = 0; i < N; i += 4) {
            for (int ii = 0; ii < 4 && i + ii < N; ii++) {
                // Convert row of A to column-major order
                for (k = 0; k < N; k++) {
                    A_col[k] = A[i+ii][k];
                }

                sum0 = _mm256_setzero_ps();
                sum1 = _mm256_setzero_ps();
                sum2 = _mm256_setzero_ps();
                sum3 = _mm256_setzero_ps();

                ap_pntr = A_col;
                bp0_pntr = B_col;
                bp1_pntr = B_col + N;
                bp2_pntr = B_col + 2 * N;
                bp3_pntr = B_col + 3 * N;

                for (k = 0; k < N; k += 32) {
                    // Unrolled loop (4 iterations)
                    Aik0 = _mm256_loadu_ps(ap_pntr);
                    Aik1 = _mm256_loadu_ps(ap_pntr + 8);
                    Aik2 = _mm256_loadu_ps(ap_pntr + 16);
                    Aik3 = _mm256_loadu_ps(ap_pntr + 24);

                    sum0 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp0_pntr), sum0);
                    sum1 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp1_pntr), sum1);
                    sum2 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp2_pntr), sum2);
                    sum3 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp3_pntr), sum3);

                    sum0 = _mm256_fmadd_ps(Aik1, _mm256_loadu_ps(bp0_pntr + 8), sum0);
                    sum1 = _mm256_fmadd_ps(Aik1, _mm256_loadu_ps(bp1_pntr + 8), sum1);
                    sum2 = _mm256_fmadd_ps(Aik1, _mm256_loadu_ps(bp2_pntr + 8), sum2);
                    sum3 = _mm256_fmadd_ps(Aik1, _mm256_loadu_ps(bp3_pntr + 8), sum3);

                    sum0 = _mm256_fmadd_ps(Aik2, _mm256_loadu_ps(bp0_pntr + 16), sum0);
                    sum1 = _mm256_fmadd_ps(Aik2, _mm256_loadu_ps(bp1_pntr + 16), sum1);
                    sum2 = _mm256_fmadd_ps(Aik2, _mm256_loadu_ps(bp2_pntr + 16), sum2);
                    sum3 = _mm256_fmadd_ps(Aik2, _mm256_loadu_ps(bp3_pntr + 16), sum3);

                    sum0 = _mm256_fmadd_ps(Aik3, _mm256_loadu_ps(bp0_pntr + 24), sum0);
                    sum1 = _mm256_fmadd_ps(Aik3, _mm256_loadu_ps(bp1_pntr + 24), sum1);
                    sum2 = _mm256_fmadd_ps(Aik3, _mm256_loadu_ps(bp2_pntr + 24), sum2);
                    sum3 = _mm256_fmadd_ps(Aik3, _mm256_loadu_ps(bp3_pntr + 24), sum3);

                    ap_pntr += 32;
                    bp0_pntr += 32;
                    bp1_pntr += 32;
                    bp2_pntr += 32;
                    bp3_pntr += 32;
                }

                // Handle remaining iterations if N is not divisible by 32
                for (; k < N; k += 8) {
                    Aik0 = _mm256_loadu_ps(ap_pntr);
                    sum0 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp0_pntr), sum0);
                    sum1 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp1_pntr), sum1);
                    sum2 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp2_pntr), sum2);
                    sum3 = _mm256_fmadd_ps(Aik0, _mm256_loadu_ps(bp3_pntr), sum3);
                    ap_pntr += 8;
                    bp0_pntr += 8;
                    bp1_pntr += 8;
                    bp2_pntr += 8;
                    bp3_pntr += 8;
                }

                C[i+ii][j] += sum256_ps(sum0);
                C[i+ii][j + 1] += sum256_ps(sum1);
                C[i+ii][j + 2] += sum256_ps(sum2);
                C[i+ii][j + 3] += sum256_ps(sum3);
            }
        }
    }
}
