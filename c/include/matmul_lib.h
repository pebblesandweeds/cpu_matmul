#ifndef MATMUL_LIB_H
#define MATMUL_LIB_H

#define N 4096
#define BLOCK_SIZE 64

void init_matrix(float matrix[N][N]);
void matmul(float A[N][N], float B[N][N], float C[N][N]);
void matmul_blocked(float A[N][N], float B[N][N], float C[N][N]);
void zero_matrix(float matrix[N][N]);

#endif
