#include "headers.h"

// ******************** General Mat-Mat Functions ******************

__global__ void gen_matvec(float *W, float *inp, float *out, float *bias, const int out_dim, const int in_dim, const int nelem_per_thread) 
{
	float* A = W;
	float* x = inp;
	float* y = out;
	float* B = bias;
	int m = out_dim;
	int n = in_dim;
	// Global and block wise thread index
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x, tId = threadIdx.x;
	// No of rows in the shared memory
	/*unsigned int smem_rows = blockDim.x * (blockIdx.x+1) * nelem_per_thread;
	if(smem_rows>=m) smem_rows = m-1;
	smem_rows = smem_rows - (blockIdx.x * blockDim.x * nelem_per_thread) + 1;
	
	unsigned int totElems = smem_rows * n, prevIndx = blockIdx.x * blockDim.x * nelem_per_thread * n;
	
	// Shared memory used combined with global memory coalescing
	*/
	extern __shared__ float s[];

	//float *As = &s[2*n];
	float *Xs = s;
	float *Bs = (s+n);

	// Filling the shared memory in a way to use memory coalescing
	for(int i = tId; i < n; i += blockDim.x ) {
		Xs[i] = x[i];
	}

	// Filling the shared memory in a way to use memory coalescing
	for(int i = tId; i < m; i += blockDim.x ) {
		Bs[i] = B[i];
	}

	/*for(int i = tId; i < totElems; i += blockDim.x ) {
		As[i] = A[prevIndx + i];
	} */

	__syncthreads();

	for(int j = 0;j<nelem_per_thread;++j){
		if(xIndex*nelem_per_thread+j >= m) break;
		float c = 0.0f;
		for(int i=0; i<n; i++)
			//c = c + Xs[i] * As[ n * (xIndex*nelem_per_thread+j) +i - prevIndx];
			c = c + Xs[i] * A[ n * (xIndex*nelem_per_thread+j) +i];
		y[xIndex*nelem_per_thread+j] = c + Bs[xIndex*nelem_per_thread+j];
	}
}
