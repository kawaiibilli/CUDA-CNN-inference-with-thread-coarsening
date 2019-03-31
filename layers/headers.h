#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define eps 1e-5

__global__ void conv1(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void conv2(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void conv3(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);

__global__ void val_checker(float* ifm, float* ofm, float *mask, int ifm_size, int ofm_size, int total_mask_size);
