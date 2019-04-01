#include "headers.h"

__device__ float get_ifm_idx(const float* ifm,int k,int i,int j,int in_h,int in_w)
{
  if(i<0 || i>=in_h || j<0 || j>=in_w)
  {
    return 0;
  }
  return ifm[k*in_h*in_w + i*in_w + j];
}
__device__ float get_ifm_plane_idx(float* ifm_plane, int i, int j, int in_h, int in_w)
{
  if(i<0 || i>=in_h || j<0 || j>=in_w)
  {
    return 0;
  }
  return ifm_plane[i*in_w + j];
}

__global__ void conv1(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity)
{
  int half_in_h = (in_h + 1)/2; 
  int half_in_w = (in_w + 1)/2;
  int out_x = blockIdx.x * blockDim.x + threadIdx.x;
  int out_y = blockIdx.y * half_in_h + threadIdx.y*granularity;
  int out_z = blockIdx.z;
  int in_x = (out_x - pad -1)*stride + mask_size;
  int in_y = (out_y - pad -1)*stride + mask_size;

  float local_mask[400];
  // __shared__ shared_mask[400];

  if((blockIdx.x == 1 && out_x >= half_in_w) || (blockIdx.y == 1 && out_y >= half_in_h))
  {
    return;
  }


  if(blockIdx.y == 0 && (out_y + granularity) > half_in_h)
  {
    granularity = half_in_h - out_y;
  }
  if(blockIdx.y == 1 && (out_y + granularity) > out_h)
  {
    granularity = out_h - out_y;
  }

  float output[32];  // max granularity = 32
  for(int g=0; g<granularity; g++)
  {
    output[g] = 0.0;
  }

  // // loading the local mask
  // for(int k=0;k<in_n;k++)
  // {
  //   for(int i=0;i<mask_size;i++)
  //   {
  //     for(int j=0;j<mask_size;j++)
  //     {
  //       local_mask[k*mask_size*mask_size + i*mask_size + j] = mask[out_z*in_n*mask_size*mask_size + k*mask_size*mask_size + i*mask_size +j];
  //     }
  //   }
  // }





  //   // loading the shared mask
  // int total_mask_elements = in_n*mask_size*mask_size;
  // int total_threads = (blockDim.x * blockDim.y);
  // int elements_per_thread = (total_mask_elements + total_threads - 1)/total_threads;
  // int block_tid = blockDim.x * threadIdx.y + threadIdx.x;
  // int mask_offset = out_z*total_mask_elements;
  // int mask_layer_size = mask_size*mask_size;
  // int curr_ele;
  // // printf("blockDim.x = %d, blockDim.y = %d, threadIdx.x = %d, threadIdx.y = %d, total_mask_elements = %d, total_threads = %d, block_tid = %d, mask_offset = %d, mask_layer_size = %d, elements_per_thread = %d\n",blockDim.x,blockDim.y,threadIdx.x, threadIdx.y, total_mask_elements,total_threads,block_tid,mask_offset,mask_layer_size,elements_per_thread);

  // for(int it=0; it<elements_per_thread; it++)
  // {
  //   curr_ele = block_tid + (it*total_threads);
  //   if(curr_ele >= total_mask_elements)
  //   {
  //     break;
  //   }
  //   shared_mask[curr_ele] = mask[mask_offset + curr_ele];
  //   // printf("succes for block_tid = %d, it = %d\n",block_tid,it );
  //   // __syncthreads(); //experiment later
  // }
  // // printf("out_z = %d\n", out_z);
  // __syncthreads(); // experiment with the syncthreads in the for loop





  // computing results
  for(int g=0; g<granularity; g++)
  {
    for(int k=0;k<in_n;k++)
    {
      for(int i=0;i<mask_size;i++)
      {
        for(int j=0;j<mask_size;j++)
        {
          output[g] += ( get_ifm_idx(ifm, k, (in_y + g*stride - i), (in_x - j), in_h, in_w) * local_mask[k*mask_size*mask_size + i*mask_size + j]);
        }
      }
    }
  }

  for(int g=0;g<granularity;g++)
  {
    ofm[out_z*out_w*out_h + (out_y + g)*out_w + out_x] = (output[g]>0)?output[g]:0;
  }
}

__device__ void load_shared_plane(const float *ifm, float *ifm_plane, int k, int in_n, int in_h, int in_w)
{
  int block_tid = blockDim.x * threadIdx.y + threadIdx.x;
  int total_threads = (blockDim.x * blockDim.y);
  int total_plane_elements = in_h*in_w;
  int elements_per_thread = (total_plane_elements + total_threads - 1)/total_threads;
  int ifm_offset = k*in_h*in_w;
  int curr_ele;

  for(int it = 0; it < elements_per_thread; it++)
  {
    curr_ele = block_tid + (it*total_threads);
    if(curr_ele >= total_plane_elements)
    {
      break;
    }
    ifm_plane[curr_ele] = ifm[ifm_offset + curr_ele];
  }
}

__global__ void conv2(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity)  // num threads in a block would be (in_w,ceil(in_h/granularity))
{

  int zflag = 0;
  int out_x = threadIdx.x;
  int out_y = threadIdx.y*granularity;
  int out_z = blockIdx.z;
  int in_x = (out_x - pad -1)*stride + mask_size;
  int in_y = (out_y - pad -1)*stride + mask_size;

  // if(out_x == 0 && out_y == 0)
  // {
  //   printf("out_z = %d,total_plane_elements = %d,elements_per_thread = %d,total_threads = %d,ifm_offset = %d\n",out_z,in_h*in_w,(in_h*in_w + blockDim.x*blockDim.y - 1)/(blockDim.x*blockDim.y),(blockDim.x * blockDim.y),out_z*in_w*in_h);
  // }

  if((out_y + granularity) > out_h)
  {
    granularity = out_h - out_y;
  }

  // float* local_mat = (float *)malloc(granularity*in_n*mask_size*mask_size * sizeof(float));
  // float local_mask[3500];
  __shared__ float shared_mask[3500];
  __shared__ float ifm_plane[3500];

  float output[32];  // max granularity = 32
  for(int g=0; g<32; g++)
  {
    output[g] = 0.0;
  }




  // // loading the local mask
  // for(int k=0;k<in_n;k++)
  // {
  //   for(int i=0;i<mask_size;i++)
  //   {
  //     for(int j=0;j<mask_size;j++)
  //     {
  //       local_mask[k*mask_size*mask_size + i*mask_size + j] = mask[out_z*in_n*mask_size*mask_size + k*mask_size*mask_size + i*mask_size +j];
  //     }
  //   }
  // }

  // loading the shared mask
  int total_mask_elements = in_n*mask_size*mask_size;
  int total_threads = (blockDim.x * blockDim.y);
  int elements_per_thread = (total_mask_elements + total_threads - 1)/total_threads;
  int block_tid = blockDim.x * threadIdx.y + threadIdx.x;
  int mask_offset = out_z*total_mask_elements;
  int mask_layer_size = mask_size*mask_size;
  int curr_ele;
  // printf("blockDim.x = %d, blockDim.y = %d, threadIdx.x = %d, threadIdx.y = %d, total_mask_elements = %d, total_threads = %d, block_tid = %d, mask_offset = %d, mask_layer_size = %d, elements_per_thread = %d\n",blockDim.x,blockDim.y,threadIdx.x, threadIdx.y, total_mask_elements,total_threads,block_tid,mask_offset,mask_layer_size,elements_per_thread);

  for(int it=0; it<elements_per_thread; it++)
  {
    curr_ele = block_tid + (it*total_threads);
    if(curr_ele >= total_mask_elements)
    {
      break;
    }
    shared_mask[curr_ele] = mask[mask_offset + curr_ele];
    // printf("succes for block_tid = %d, it = %d\n",block_tid,it );
    // __syncthreads(); //experiment later
  }
  // printf("out_z = %d\n", out_z);
  __syncthreads(); // experiment with the syncthreads in the for loop






  // ///////////////// debug : load the corresponding layer 
  // load_shared_plane(ifm, ifm_plane, out_z, in_n, in_h, in_w);
  // __syncthreads();

  // ofm[out_z*out_h*out_w + threadIdx.y*out_w + threadIdx.x ] = get_ifm_plane_idx(ifm_plane, out_y, out_x, in_h, in_w);
  // ///////////////////////////////////

  // loading the local ifm
  // for(int g=0; g<granularity && (out_y + g)<out_h; g++)
  // {
  //   for(int k=0; k<in_n; k++)
  //   {
  //     for(int i=0; i<mask_size; i++)
  //     {
  //       for(int j=0; j<mask_size; j++)
  //       {
  //         local_mat[g*total_mask_elements + k*mask_size*mask_size + i*mask_size + j] = 0; //(((in_y + g*stride - i)>=0 && (in_y + g*stride - i) < in_h && (in_x - j)>=0 && (in_x - j) < in_w) ? ifm[k*in_h*in_w + (in_y + g*stride - i)*in_w + (in_x - j)] : 0);
  //       }
  //     }
  //   }
  // }
  


  // //computing output values
  // for(int g=0; g<granularity; g++)
  // {
  //   for(int k=0; k<in_n; k++)
  //   {
  //     for(int i=0; i<mask_size; i++)
  //     {
  //       for(int j=0; j<mask_size; j++)
  //       {
  //         // output[g] += (mask[out_z*in_n*mask_size*mask_size + k*mask_layer_size + i*mask_size + j] * get_ifm_idx(ifm, k, (in_y + g*stride - i), (in_x - j), in_h, in_w));
  //         output[g] +=  get_ifm_idx(ifm, k, (in_y + g*stride - i), (in_x - j), in_h, in_w) * local_mask[k*mask_size*mask_size + i*mask_size + j];
  //         // output[g] +=  get_ifm_idx(ifm, k, (in_y + g*stride - i), (in_x - j), in_h, in_w) * shared_mask[k*mask_size*mask_size + i*mask_size + j];
  //         // if(get_ifm_idx(ifm, k, (in_y + g*stride - i), (in_x - j), in_h, in_w)> eps)
  //         // {
  //         //   zflag = 1;, 
  //         // }
  //       }
  //     }
  //   }
  // }



  //computing output values by accumulating
  for(int k=0; k<in_n; k++)
  {
    load_shared_plane(ifm, ifm_plane, k, in_n, in_h, in_w);
    __syncthreads();
    for(int g=0; g<granularity; g++)
    {
      for(int i=0;i<mask_size;i++)
      {
        for(int j=0;j<mask_size;j++)
        {
          // output[g] +=  get_ifm_plane_idx(ifm_plane, (in_y + g*stride - i), (in_x - j), in_h, in_w) * local_mask[k*mask_size*mask_size + i*mask_size + j];
          output[g] +=  get_ifm_plane_idx(ifm_plane, (in_y + g*stride - i), (in_x - j), in_h, in_w) * shared_mask[k*mask_size*mask_size + i*mask_size + j];
        }
      }
    }
  }



  // // printf("out_z = %d, out_w = %d, out_h = %d, out_x = %d, out_y = %d, granularity = %d \n",out_z,out_w,out_h,out_x,out_y,granularity);
  // writing back to global memory
  for(int g=0;g<granularity;g++)
  {
    ofm[out_z*out_w*out_h + (out_y + g)*out_w + out_x] = (output[g]>0)?output[g]:0;
  }
}

__device__ void load_mask_planes(float *mask, float *mask_planes, int k, int start_layer, int mask_size, int granularity, int in_n, int out_m)
{
  int block_tid = blockDim.x * threadIdx.y + threadIdx.x;
  if(block_tid>=(mask_size*mask_size))
    {
      return;
    }

  int offset = k*mask_size*mask_size;
  int curr_ele, out_idx;
  for(int g=0;g<granularity; g++)
  {
    curr_ele = offset + (start_layer + g)*in_n*mask_size*mask_size + block_tid;
    out_idx = g*mask_size*mask_size + block_tid;
    mask_planes[out_idx] = mask[curr_ele];
  }
}


__global__ void conv3(const float *ifm, float *ofm, float *mask, int in_h, int in_w, int in_n, int out_h, int out_w, int out_m, int mask_size, int pad, int stride, int granularity)  // num threads in a block would be (in_w,ceil(in_h/granularity))
{
  int zflag = 0;

  int out_x = threadIdx.x;
  int out_y = threadIdx.y;
  int out_z = blockIdx.z*granularity;

  int in_x = (out_x - pad -1)*stride + mask_size;
  int in_y = (out_y - pad -1)*stride + mask_size;

  if((out_z + granularity) > out_m)
  {
    granularity = out_m - out_z;
  }

  __shared__ float mask_planes[3500];
  __shared__ float ifm_plane[3030];

  float output[32];  // max granularity = 32
  for(int g=0; g<granularity; g++)
  {
    output[g] = 0.0;
  }

  //computing output values by accumulating
  for(int k=0; k<in_n; k++)
  {
    load_shared_plane(ifm, ifm_plane, k, in_n, in_h, in_w);
    
    __syncthreads();
    load_mask_planes(mask, mask_planes, k, out_z, mask_size, granularity, in_n, out_m);

    __syncthreads();

    for(int g=0; g<granularity; g++)
    {
      for(int i=0;i<mask_size;i++)
      {
        for(int j=0;j<mask_size;j++)
        {
          output[g] +=  get_ifm_plane_idx(ifm_plane, (in_y - i), (in_x - j), in_h, in_w) * mask_planes[g*mask_size*mask_size + i*mask_size + j];
        }
      }
    }
  }

  // // printf("out_z = %d, out_w = %d, out_h = %d, out_x = %d, out_y = %d, granularity = %d \n",out_z,out_w,out_h,out_x,out_y,granularity);
  // // writing back to global memory
  for(int g=0;g<granularity;g++)
  {
    ofm[(out_z + g)*out_w*out_h + out_y*out_w + out_x] = (output[g]>0)?output[g]:0;
  }
}



__global__ void val_checker (float* ifm, float* ofm, float *mask, int ifm_size, int ofm_size, int total_mask_size )
{
  int zflag = 0;
  for(int i=0;i<ifm_size;i++)
  {
    if(fabs(ifm[i])>eps)
    {
      zflag = 1;
    }
  }
  if(zflag==0)
  {
    printf("all ifm elements zero\n");
  }
  __syncthreads();

  zflag = 0;
  for(int i=0;i<ofm_size;i++)
  {
    if(fabs(ofm[i])>eps)
    {
      zflag = 1;
    }
  }
  if(zflag==0)
  {
    printf("all ofm elements zero\n");
  }
  __syncthreads();

  zflag = 0;
  for(int i=0;i<total_mask_size;i++)
  {
    if(fabs(mask[i])>eps)
    {
      zflag = 1;
    }
  }
  if(zflag==0)
  {
    printf("all mask elements zero\n");
  }
  __syncthreads();
}
