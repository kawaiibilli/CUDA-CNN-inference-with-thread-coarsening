#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void gran_pooling(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity);
__global__ void shared_pool(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity);

