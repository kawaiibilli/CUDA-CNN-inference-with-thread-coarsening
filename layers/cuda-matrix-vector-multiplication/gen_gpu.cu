#include "gen_gpu.h"

// ******************** General Mat-Vec Functions ******************


//   y = Ax+B
__global__ void gen_matvec(float *A, float *x, float *y, float *B, const int m, const int n, const int nelem_per_thread) 
{
	// Global and block wise thread index
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x, tId = threadIdx.x;
	
	extern __shared__ float s[];

	
	float *Xs = s;
	float *Bs = (s+n);

	// Filling the shared memory in a way to use memory coalescing
	for(int i = tId; i < n; i += blockDim.x ) {
		Xs[i] = x[i];
	}

	// Filling the shared memory in a way to use memory coalescing
	/*for(int i = tId; i < m; i += blockDim.x ) {
		Bs[i] = B[i];
	}*/

	__syncthreads();

	for(int j = 0;j<nelem_per_thread;++j){
		if(xIndex*nelem_per_thread+j >= m) break;
		float c = 0.0f;
		for(int i=0; i<n; i++)
			//c = c + Xs[i] * As[ n * (xIndex*nelem_per_thread+j) +i - prevIndx];
			c = c + Xs[i] * A[ n * (xIndex*nelem_per_thread+j) +i];
		//y[xIndex*nelem_per_thread+j] = c + Bs[xIndex*nelem_per_thread+j];
		y[xIndex*nelem_per_thread+j] = c + B[xIndex*nelem_per_thread+j];
	}
}




//   y = Ax+B
__global__ void gen_matvec_noshared(float *A, float *x, float *y, float *B, const int m, const int n, const int nelem_per_thread) 
{
	// Global and block wise thread index
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;	

	for(int j = 0;j<nelem_per_thread;++j){
		if(xIndex*nelem_per_thread+j >= m) break;
		float c = 0.0f;
		for(int i=0; i<n; i++)
			//c = c + Xs[i] * As[ n * (xIndex*nelem_per_thread+j) +i - prevIndx];
			c = c + x[i] * A[ n * (xIndex*nelem_per_thread+j) +i];
		y[xIndex*nelem_per_thread+j] = c + B[xIndex*nelem_per_thread+j];
	}
}



//   y = Ax+B
__global__ void gen_matvec_nocoarse(float *A, float *x, float *y, float *B, const int m, const int n) 
{
	// Global and block wise thread index
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x, tId = threadIdx.x;
	
	extern __shared__ float s[];

	
	float *Xs = s;
	float *Bs = (s+n);

	// Filling the shared memory in a way to use memory coalescing
	for(int i = tId; i < n; i += blockDim.x ) {
		Xs[i] = x[i];
	}

	// Filling the shared memory in a way to use memory coalescing
	/*for(int i = tId; i < m; i += blockDim.x ) {
		Bs[i] = B[i];
	}*/

	__syncthreads();

	if(xIndex < m) {
		float c = 0.0f;
		for(int i=0; i<n; i++)
			c = c + Xs[i] * A[ n * (xIndex) +i];
		//y[xIndex] = c + Bs[xIndex];
		y[xIndex] = c + B[xIndex];
	}
}


//   y = Ax+B
__global__ void gen_matvec_nocoarse_noshared(float *A, float *x, float *y, float *B, const int m, const int n) 
{
	// Global and block wise thread index
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(xIndex < m){
		float c = 0.0f;
		for(int i=0; i<n; i++)
			c = c + x[i] * A[ n * (xIndex) +i];
		y[xIndex] = c + B[xIndex];
	}
}

