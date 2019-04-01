#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream> 

#define eps 1e-5

class FCLayer {
public:
	std::string name;
	int inputs;
	int outputs;
	float* weights;	// dim - outputs x inputs 
	float* biases;
	FCLayer(std::string name, int inputs, int outputs);
	FCLayer();
};

class ConvLayer {
public:
	std::string name;
	int filter_size;
	int num_layers; 
	int depth;
	float* weights;	// dim - filter_size x filter_size x num_layers x depth
	float* biases;
	ConvLayer();
	ConvLayer(std::string name, int filter_size, int num_layers, int depth);
};


__global__ void conv1_kernel(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void conv2_kernel(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void conv3_kernel(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void val_checker(float* ifm, float* ofm, float *mask, int ifm_size, int ofm_size, int total_mask_size);

__global__ void gran_pooling(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity);

__global__ void shared_pool(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity);

__global__ void gen_matvec(float *A, float *x, float *y, float *B, const int m, const int n, const int nelem_per_thread);

__host__ FCLayer processFC(std::stringstream& ss) ;

__host__ ConvLayer processConv(std::stringstream& ss);
