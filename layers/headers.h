#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void l1_conv2d(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_w, int mask_size, int pad, int stride, int granularity);

__global__ void gen_conv2d(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_w, int out_m, int mask_size, int pad, int stride, int granularity);
