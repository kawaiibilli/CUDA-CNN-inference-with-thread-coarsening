#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ******************** General Mat-Mat Functions ******************

__global__ void gen_matvec(float *W, float *inp, float *out, float *bias, const int out_dim, const int in_dim, const int nelem_per_thread);
