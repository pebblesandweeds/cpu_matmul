#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include "../include/matmul_lib.h"
#include "../include/time_utils.h"
#include "../include/check_utils.h"

int main() {
	float (*A)[N] = (float(*)[N])_mm_malloc(sizeof(float[N][N]), 32);
    float (*B)[N] = (float(*)[N])_mm_malloc(sizeof(float[N][N]), 32);
    float (*C_naive)[N] = (float(*)[N])_mm_malloc(sizeof(float[N][N]), 32);
    float (*C_scalar)[N] = (float(*)[N])_mm_malloc(sizeof(float[N][N]), 32);
    float (*C_vectorized)[N] = (float(*)[N])_mm_malloc(sizeof(float[N][N]), 32);

     if (A == NULL || B == NULL || C_naive == NULL || C_scalar == NULL || C_vectorized == NULL) {   
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));

    init_matrix(A);
    init_matrix(B);
    zero_matrix(C_naive);
    zero_matrix(C_scalar);
    zero_matrix(C_vectorized);

    // See Python Numpy code comments for FLOP calculation explanation
    long long flop = 2LL * N * N * N;
    printf("Total FLOP: %lld\n", flop);
    printf("%.2f GFLOP\n", flop / 1e9);

    // Naive matmul 
    double time_naive = timed_matmul(matmul, A, B, C_naive);
    printf("Naive matmul time taken: %.6f seconds\n", time_naive);
    double gflops_naive = (flop / time_naive) / 1e9;
    printf("Naive matmul: %.2f GFLOPS\n", gflops_naive);

    // Matmul with scalars 
    double time_scalar = timed_matmul(matmul_scalar, A, B, C_scalar);
    printf("Scalar matmul time taken: %.6f seconds\n", time_scalar);
    double gflops_scalar = (flop / time_scalar) / 1e9;
    printf("Scalar matmul: %.2f GFLOPS\n", gflops_scalar);

    // Check matrices
    bool matrices_match_naive_scalar = check_matrices(C_naive, C_scalar, 1e-5);
    if (matrices_match_naive_scalar) {    
        printf("Matrices match within tolerance.\n");
    } else {
        printf("Matrices do not match within tolerance.\n");
    }

    // Matmul with vectors 
    double time_vectorized = timed_matmul(matmul_vectorized, A, B, C_vectorized);
    printf("Vectorized matmul time taken: %.6f seconds\n", time_vectorized);
    double gflops_vectorized = (flop / time_vectorized) / 1e9;
    printf("Vectorized matmul: %.2f GFLOPS\n", gflops_vectorized);

    // Check matrices
    bool matrices_match_naive_vectorized = check_matrices(C_naive, C_vectorized, 1e-3);
    if (matrices_match_naive_vectorized) {    
        printf("Matrices match within tolerance.\n");
    } else {
        printf("Matrices do not match within tolerance.\n");
    }

	_mm_free(A);
    _mm_free(B);
    _mm_free(C_naive);
    _mm_free(C_scalar);
    _mm_free(C_vectorized);	

    return 0;
}
