#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ******************** General Mat-Vec Functions ******************


// y = Ax+B

__global__ void gen_matvec(float *A, float *x, float *y, float *B, const int m, const int n, const int nelem_per_thread);
__global__ void gen_matvec_noshared(float *A, float *x, float *y, float *B, const int m, const int n, const int nelem_per_thread);
__global__ void gen_matvec_nocoarse(float *A, float *x, float *y, float *B, const int m, const int n);  
__global__ void gen_matvec_nocoarse_noshared(float *A, float *x, float *y, float *B, const int m, const int n);
