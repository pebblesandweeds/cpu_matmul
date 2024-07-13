#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "../include/matmul_lib.h"
#include "../include/time_utils.h"

int main() {
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));

    init_matrix(A);
    init_matrix(B);
    zero_matrix(C);

    // See Python Numpy code comments for FLOP calculation explanation
    long long flop = 2LL * N * N * N;
    printf("Total FLOP: %lld\n", flop);
    printf("%.2f GFLOP\n", flop / 1e9);

    // Naive matmul 
    double time_naive = timed_matmul(matmul, A, B, C);
    printf("Naive matmul time taken: %.6f seconds\n", time_naive);
    double gflops_naive = (flop / time_naive) / 1e9;
    printf("Naive matmul: %.2f GFLOPS\n", gflops_naive);

    // Matmul with blocks
    zero_matrix(C);

    double time_blocked = timed_matmul(matmul_blocked, A, B, C);
    printf("Blocked matmul time taken: %.6f seconds\n", time_blocked);
    double gflops_blocked = (flop / time_blocked) / 1e9;
    printf("Blocked matmul: %.2f GFLOPS\n", gflops_blocked);

    free(A);
    free(B);
    free(C);

    return 0;
}
