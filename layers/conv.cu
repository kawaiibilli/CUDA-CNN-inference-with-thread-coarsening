#include "headers.h"


__global__ void l1_conv2d(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_w, int out_m, int mask_size, int pad, int stride, int granularity)
{

  float local_mat[in_n][mask_size][mask_size];
  float local_mask[in_n][mask_size][mask_size];
  int out_x = blockIdx.x*blockDim.x + threadIdx.x;
  int out_y = blockIdx.y*blockDim.y + threadIdx.y;
  int out_z = blockIdx.z;
  int in_x = (out_x - pad -1)*stride + mask_size;
  int in_y = (out_y - pad -1)*stride + mask_size;

 float output = 0.0;

 for(int k=0; k<in_n;k++)
 {
  for(int i=0; i<mask_size; i++)
  {
    for(int j=0; j<mask_size;j++)
    {
      local_mat[k][i][j] = (((in_y-i)>=0 && (in_y-i)<in_h && (in_x-j)>=0 && (in_x-j)<in_w ) ? ifm[k*in_h*in_w + (in_y - i)*in_w + (in_x - j)] : 0);     // ifm must be stored in layer # X row X column format
    }
  }
 }

 for(int k=0; k<in_n;k++)
 {
  for(int i=0; i<mask_size; i++)
  {
    for(int j=0; j<mask_size;j++)
    {
      local_mask[k][i][j] = mask[k*mask_size*mask_size + i*mask_size + j];
    }
  }
 }


 for(int k=0;k<in_n;k++)
 {
  for(int i=0;i<mask_size;i++)
  {
    for(int j=0;j<mask_size;j++)
    {
      output += local_mat[k][i][j] * local_mask[k][i][j];
    }
  }
 }
 ofm[out_z*out_w*out_w + out_y*out_w + out_x] = output;

}

__global__ void gen_conv2d(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_w, int out_m, int mask_size, int pad, int stride, int granularity)  // num threads in a block would be (in_w,ceil(in_h/granularity))
{

  int out_x = threadIdx.x;
  int out_y = threadIdx.y;
  int out_z = blockIdx.z;
  int in_x = (out_x - pad -1)*stride + mask_size;
  int in_y = (out_y - pad -1)*stride + mask_size;

  if(out_y + granularity>=out_w)
  {
    granularity = out_w - out_y;
  }

  __shared__ float local_mat[granularity][in_n][mask_size][mask_size];
  __shared__ float local_mask[in_n][mask_size][mask_size];
  float output[granularity];
  for(int g=0;g<granularity;g)
  {
    output[g] = 0.0;
  }

  for(int g=0;g<granularity && (out_y + g)< ;g++)
  {
    for(int k=0;k<in_n;k++)
    {
      local_mat[g][k][out_y][out_x] = (((in_y-i)>=0 && (in_y-i)<in_h && (in_x-j)>=0 && (in_x-j)<in_w) ? ifm[k*in_h*in_w + (in_y + g - i)*in_w + (in_x - j)] : 0);
      // __syncthreads();  // maybe ?: experiment for better speedup
    }
  }
  
  __syncthreads(); // maybe ?

  if(out_x<mask_size && out_y<mask_size)
  {
    for(int k=0;k<in_n;k++)
    {
      local_mask[k][out_y][out_x] = mask[k*mask_size*mask_size + out_y*mask_size + out_x];

      // __syncthreads(); //experiment later 
    }
  }

  __syncthreads(); // experiment with the syncthreads in the for loop

  for(int g=0;g<granularity;g++)
  {
    for(int k=0;k<in_n;k++)
    {
      for(int i=0;i<mask_size;i++)
      {
        for(int j=0;j<mask_size;j++)
        {
          output[g] += local_mask[k][i][j] * local_mat[g][k][out_y][out_x];
        }

      }
    }
  }

  for(int g=0;g<granularity;g++)
  {
    ofm[out_z*out_w*out_w + (out_y + g)*out_w + out_x] = output[g];
  }
}