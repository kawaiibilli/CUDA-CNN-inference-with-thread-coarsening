#include "gen_gpu.h"

// ******************** General Mat-Mat Functions ******************

__global__ void gen_matvec(float *A, float *x, float *y, const int m, const int n, const int nelem_per_thread) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex+nelem_per_thread-1 < m ){
    for(int j = 0;j<nelem_per_thread;++j){
      float c = 0.0f;
      for(int i=0; i<n; i++)
        c = c + x[i] * A[ m * (xIndex*nelem_per_thread+j) +i];
      y[xIndex+j] = c;
    }
  }
}












