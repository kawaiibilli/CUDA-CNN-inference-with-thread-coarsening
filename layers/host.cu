#include "headers.h"

int main(void)
{
	cudaError_t err = cudaSuccess;
    int num_in_fm = 96;
   	int in_fm_h = 55;
   	int in_fm_w = 55;
   	int num_out_fm = 96;
   	int out_fm_w = 27;
   	int out_fm_h = 27;
   	int mask_size = 3;
   	int stride = 2;
   	int granularity = 1;

   	int in_size = num_in_fm*in_fm_w*in_fm_h*sizeof(float);
   	float *h_ifm = malloc(in_size);
   	int out_size = num_out_fm*out_fm_w*out_fm_w*sizeof(float);
   	float *h_ofm = malloc(out_size);

   	for(int k=0;k<num_in_fm;k++)
   	{
   		for(int i=0;i<in_fm_h;i++)
   		{
   			for(int j=0;j<in_fm_w;j++)
   			{
   				h_ifm[k*in_fm_w*in_fm_h + i*in_fm_w + j] = rand()/(float) RAND_MAX;
   			}
   		}
   	}

	for(int k=0;k<num_out_fm;k++)
   	{
   		for(int i=0;i<out_fm_h;i++)
   		{
   			for(int j=0;j<out_fm_w;j++)
   			{
   				h_ofm[k*out_fm_w*out_fm_h + i*out_fm_w + j] = rand()/(float) RAND_MAX;
   			}
   		}
   	}

   	float *d_ifm = NULL;
    err = cudaMalloc((void **)&d_ifm, in_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device ifm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_ofm = NULL;
    err = cudaMalloc((void **)&d_ofm, out_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device ofm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_ifm, h_ifm, in_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix ifm from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_ofm, h_ofm, out_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix ofm from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    dim3 blocksPerGrid( num_out_fm,1,1);
    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity -1)/granularity) ,1);

    gen_conv2d <<<blocksPerGrid, threadsPerBlock>>>(d_ifm,d_ofm, );

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel gen_conv2d (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


}