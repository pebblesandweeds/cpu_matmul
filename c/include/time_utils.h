#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include "matmul_lib.h"

double get_time();
double timed_matmul(void (*matmul_func)(float[N][N], float[N][N], float[N][N]), float A[N][N], float B[N][N], float C[N][N]);

#endif
