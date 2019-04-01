#include "headers.h"

int main(void)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float delta = 0.0; //to measure time
	cudaError_t err = cudaSuccess;

    // int num_in_fm = 3;
   	// int in_fm_h = 227;
   	// int in_fm_w = 227;
   	// int num_out_fm = 96;
   	// int out_fm_w = 55;
   	// int out_fm_h = 55;
   	// int mask_size = 11;
   	// int stride = 4;
   	// int granularity = 4;
   	// int pad = 0;

	int num_in_fm = 96;
   	int in_fm_h = 27;
   	int in_fm_w = 27;
   	int num_out_fm = 256;
   	int out_fm_w = 27;
   	int out_fm_h = 27;
   	int mask_size = 5;
   	int stride = 1;
   	int granularity = 1;
   	int pad = 2;

   	// printf("granularity = %d\n",granularity);
   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
   	float *h_ifm = (float*) malloc(in_size);
   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
   	float *h_ofm = (float *)malloc(out_size);
   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
   	float *h_mask = (float *)malloc(total_mask_size);
   	float *test_ofm = (float *) malloc(out_size);

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

   	for(int k1 =0;k1<num_out_fm;k1++)
   	{
   		for(int k2=0;k2<num_in_fm;k2++)
   		{
   			for(int i=0;i<mask_size;i++)
   			{
   				for(int j=0;j<mask_size;j++)
   				{
   					h_mask[k1*num_in_fm*mask_size*mask_size + k2*mask_size*mask_size + i*mask_size + j] = rand()/(float) RAND_MAX;
   				}
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

    float *d_mask = NULL;
    err = cudaMalloc((void **)&d_mask, total_mask_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_ifm, h_ifm, in_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix ifm from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // dim3 blocksPerGrid(num_out_fm,2,2);
    // dim3 threadsPerBlock((out_fm_w + 1)/2, (((out_fm_h+1)/2 + granularity -1)/granularity) ,1);
    dim3 blocksPerGrid(num_out_fm,1,1);
    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);



    /*
 //    ////////// shared ifm debugging////////////////////////////////////

	// blocksPerGrid.x = 1;
 // 	blocksPerGrid.y = 1;
 // 	blocksPerGrid.z =  num_out_fm;

 // 	threadsPerBlock.x = out_fm_w;
 // 	threadsPerBlock.y = out_fm_h;
 // 	threadsPerBlock.z = 1;


 //    conv2<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, 1);
	    
	// err = cudaGetLastError();
	// if (err != cudaSuccess)
	// {
	// 	fprintf(stderr, "Failed to launch kernel conv2 (error code %s)!\n", cudaGetErrorString(err));
	// 	exit(EXIT_FAILURE);
	// }

	// ///////////////////////////////////////////////////////////////////

    */




	// // for conv1
 //    for(int g=1;g<=16;g++)
 //    {
 //    	blocksPerGrid.x = 2;
 //    	blocksPerGrid.y = 2;
 //    	blocksPerGrid.z =  num_out_fm;

 //    	threadsPerBlock.x = (out_fm_w + 1)/2;
 //    	threadsPerBlock.y = (((out_fm_h+1)/2 + g -1)/g);
 //    	threadsPerBlock.z = 1;

	// 	printf("threadsPerBlock = %d,%d,%d\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	// 	printf("blocksPerGrid = %d,%d,%d\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);

	// 	cudaEventRecord(start);
	//     conv1<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, g);
	//     cudaEventRecord(stop);
	//     cudaEventSynchronize(stop);
	// 	delta = 0;
	// 	cudaEventElapsedTime(&delta, start, stop);
	// 	printf("granularity = %d, time in milliseconds = %f\n",g,delta);
	//     // conv1<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
	//     // val_checker<<<1,1>>>(d_ifm,d_ofm,d_mask,num_in_fm*in_fm_w*in_fm_h , num_out_fm*out_fm_w*out_fm_h, num_out_fm*num_in_fm*mask_size*mask_size);
	//     err = cudaGetLastError();
	//     if (err != cudaSuccess)
	//     {
	//         fprintf(stderr, "Failed to launch kernel conv2 (error code %s)!\n", cudaGetErrorString(err));
	//         exit(EXIT_FAILURE);
	//     }
	// }
	




	// for conv2
    for(int g=1;g<=13;g++)
    {
    	blocksPerGrid.x = 1;
    	blocksPerGrid.y = 1;
    	blocksPerGrid.z =  num_out_fm;

    	threadsPerBlock.x = out_fm_w;
    	threadsPerBlock.y = ((out_fm_h + g - 1)/g);
    	threadsPerBlock.z = 1;

		printf("threadsPerBlock = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
		cudaEventRecord(start);

	    conv2<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, g);
	    
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
		delta = 0;
		cudaEventElapsedTime(&delta, start, stop);
		printf("granularity = %d, time in milliseconds = %f\n",g,delta);

	    err = cudaGetLastError();
	    if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to launch kernel conv2 (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
	}


	// // for conv3
	// for(int g=1;g<=16;g++)
 //    {
 //    	blocksPerGrid.x = 1;
 //    	blocksPerGrid.y = 1;
 //    	blocksPerGrid.z =  (num_out_fm + g - 1)/g;

 //    	threadsPerBlock.x = out_fm_w;
 //    	threadsPerBlock.y = out_fm_h;
 //    	threadsPerBlock.z = 1;

	// 	printf("threadsPerBlock = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	// 	printf("blocksPerGrid = %d,%d,%d\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);

	// 	cudaEventRecord(start);

	//     conv3<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, g);

	//     cudaEventRecord(stop);
	//     cudaEventSynchronize(stop);
	// 	delta = 0;
	// 	cudaEventElapsedTime(&delta, start, stop);
	// 	printf("granularity = %d, time in milliseconds = %f\n",g,delta);
	//     // conv1<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
	//     // val_checker<<<1,1>>>(d_ifm,d_ofm,d_mask,num_in_fm*in_fm_w*in_fm_h , num_out_fm*out_fm_w*out_fm_h, num_out_fm*num_in_fm*mask_size*mask_size);
	//     err = cudaGetLastError();
	//     if (err != cudaSuccess)
	//     {
	//         fprintf(stderr, "Failed to launch kernel conv3 (error code %s)!\n", cudaGetErrorString(err));
	//         exit(EXIT_FAILURE);
	//     }
	// }

    err = cudaMemcpy(h_ofm, d_ofm, out_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix ofm from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("printing 10 ofm elements\n");
    for(int i=30;i<40;i++)
    {
        printf("ofm[%d] = %f\n",i,h_ofm[i]);
    }


 //    printf("debugging : \n");
 //    printf("printing 10 ifm elements\n");

 //    for(int i=169;i<179;i++)
 //    {
 //        printf("ifm[%d] = %f\n",i,h_ifm[i]);
 //    }

	// for(int i=0;i<num_in_fm*in_fm_w*in_fm_w;i++)
 //    {
 //        if(fabs(h_ofm[i] - h_ifm[i])>eps)
 //        {
 //        	printf("not matching at i=%d\n",i);
 //        	exit(EXIT_FAILURE);
 //        }
 //    }    
 //    printf("all matched !! \n");



    int zflag = 0;
    for(int i=0;i< (num_out_fm*out_fm_h*out_fm_w); i++)
    {
        if(fabs(h_ofm[i])>eps)
        {
			zflag = 1;
        }
    }
    if(zflag==0)
    {
        printf("all elements zero\n");
    }

    printf("Verifying results \n");
    for(int i=0;i<num_out_fm*out_fm_w*out_fm_h;i++)
    {
    	test_ofm[i] = 0.0;
    }

    for(int l=0; l < num_out_fm; l++)
    {
		for(int i=0; i < out_fm_h; i++)
		{
			for(int j=0; j<out_fm_w; j++)
			{
				int in_i = (i - pad - 1)*stride + mask_size;
				int in_j =  (j - pad -1)*stride + mask_size;
				float tmp = 0.0;
				for(int k=0;k<num_in_fm;k++)
				{
					for(int ii = 0; ii<mask_size; ii++)
					{
						for(int jj = 0; jj<mask_size; jj++)
						{
							float tt = (((in_i - ii)>=0 && (in_i - ii)<in_fm_h && (in_j - jj)>=0 && (in_j - jj) < in_fm_w ) ? h_ifm[k*in_fm_h*in_fm_w + (in_i - ii)*in_fm_w + (in_j - jj)] : 0);
							tmp += (h_mask[l*num_in_fm*mask_size*mask_size + k*mask_size*mask_size + ii*mask_size + jj]*tt);
						}
					}
				}
				test_ofm[l*out_fm_h*out_fm_w + i*out_fm_w + j] = tmp;
      		}
    	}
    	// test_ofm[l*out_fm_h*out_fm_w + i*out_fm_w + j] = (test_ofm[l*out_fm_h*out_fm_w + i*out_fm_w + j]>0) ? test_ofm[l*out_fm_h*out_fm_w + i*out_fm_w + j] : 0;
  	}

  	printf("printing 10 test_ofm elements\n");
  	for(int i=30;i<40;i++)
  	{
  		printf("test_ofm[%d] = %f\n",i,test_ofm[i]);
  	}

  	for(int i=0;i<num_out_fm*out_fm_w*out_fm_h;i++)
  	{
  		if(fabs(test_ofm[i] - h_ofm[i]) > eps)
  		{
  			fprintf(stderr, "Result verification failed at element (%d) !\n",i);
    		exit(EXIT_FAILURE);
  		}
  	}

  	printf("all elements matched !!\n");


    err = cudaFree(d_ifm);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_ofm);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix ofm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_mask);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_ifm);
    free(h_ofm);
    free(h_mask);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Done\n");
    return 0;

}