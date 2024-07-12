#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N 4096 

// Similar to Numpy but this function will generate a uniform distribution between -1 and 1
// Numpy in Python will generate a normal distribution (bell curve centered at 0)
// If we want a normal distribution in C we would need to change this function
void init_matrix(float matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
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

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

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

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    // See Python Numpy code comments for FLOP calculation explanation
    long long flop = 2LL * N * N * N;
    printf("Total FLOP: %lld\n", flop);
    printf("%.2f GFLOP\n", flop / 1e9);

    double st = get_time();
    matmul(A, B, C);
    double et = get_time();
    double s = et - st; 

    printf("Time taken: %.6f seconds\n", s);

    double gflops = (flop / s) / 1e9;
    printf("%.2f GFLOPS\n", gflops);

    free(A);
    free(B);
    free(C);

    return 0;
}
