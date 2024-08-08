#ifndef MATMUL_LIB_H
#define MATMUL_LIB_H

#define N 8192 
#define BLOCK_SIZE 64
#define TILE_SIZE 32 
#define UNROLL_FACTOR 4

void init_matrix(float matrix[N][N]);
void matmul(float A[N][N], float B[N][N], float C[N][N]);
void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]);
void zero_matrix(float matrix[N][N]);
void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]);
void AddDot(int k, float *x, int incx, float *y, float *gamma);

#endif
