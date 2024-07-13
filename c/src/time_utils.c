#include "../include/time_utils.h"
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

double timed_matmul(void (*matmul_func)(float[N][N], float[N][N], float[N][N]), float A[N][N], float B[N][N], float C[N][N]) {
    double start_time = get_time();
    matmul_func(A, B, C);
    double end_time = get_time();
    return end_time - start_time;
}
