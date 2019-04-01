#include <vector>
#include <iostream>
#include <fstream>
#include <locale>
#include <limits>
#include <sstream>
#include <algorithm>
#include "headers.h"


int main(){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	float delta = 0.0; //to measure time
	cudaError_t err = cudaSuccess;
	std::string line;
	std::ifstream in("alexnet.csv");

	float *d_ofm, *d_out, *d_ofm_3, *d_ofm_4, *d_ofm_5, *out_1, *out_2, *out_3;
	ConvLayer conv_1, conv2, conv3, conv4, conv5;
	FCLayer fc6 , fc7, fc8;
	int granularity = 1;
	printf(" start \n");
	while(getline(in, line)) {
		std::stringstream lineStream(line);
    	if (line[0] == 'c') { //Convolutional layer 
    		ConvLayer c = processConv(lineStream);
    		printf("%s c.name %d \n",c.name.c_str(),c.filter_size);
    		if(c.name == "conv1"){ conv_1 = c;}
    		if(c.name == "conv2"){ conv2 = c;}
    		if(c.name == "conv3"){ conv3 = c;}
    		if(c.name == "conv4"){ conv4 = c;}
    		if(c.name == "conv5"){ conv5 = c;}
    	} else if (line[0] == 'f') { //Fully Connected Layer
    		FCLayer f = processFC(lineStream);
    		if(f.name == "fc6"){ fc6 = f;}
    		if(f.name == "fc7"){ fc7 = f;}
    		if(f.name == "fc8"){ fc8 = f;}
    	}
    }
    printf("load complete \n");
	for(granularity =1; granularity <= 16; granularity++) {
		// Convolution layer 1
		if(true){
	
			// i/p : 3x227x227, o/p : 96x55x55, filter : 11x11x96x3
			int num_in_fm = 3;
		   	int in_fm_h = 227;
		   	int in_fm_w = 227;
		   	int num_out_fm = 96;
		   	int out_fm_w = 55;
		   	int out_fm_h = 55;
		   	int mask_size = 11;
		   	int stride = 4;
		   	int pad = 0;
		   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
		   	printf("57\n");
		   	float *h_ifm = (float*) malloc(in_size);
		   	printf("59\n");
		   	// random generation of the i/p image matrix 
		   	for(int i=0;i< num_in_fm*in_fm_w*in_fm_h;i++){
		   		h_ifm[i] = rand()/(float) RAND_MAX;
		   	}
		   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
		   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
		   	printf(" In conv 1 \n");
			printf(" filter_size : %d , num_layers : %d, depth : %d \n",conv_1.filter_size,conv_1.num_layers,conv_1.depth);
	
		   	float *d_ifm = NULL;
		    err = cudaMalloc((void **)&d_ifm, in_size);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device ifm (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
						printf("78\n");

		    d_ofm = NULL;
		    err = cudaMalloc((void **)&d_ofm, out_size);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device ofm (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
						printf("87\n");

		    float *d_mask = NULL;
		    err = cudaMalloc((void **)&d_mask, total_mask_size);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device mask (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
				printf("96\n");

		    err = cudaMemcpy(d_ifm, h_ifm, in_size, cudaMemcpyHostToDevice);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to copy matrix ifm from host to device (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
			printf("101\n");
			float *h_mask = conv_1.weights;
			err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
				printf("108\n");

		
		    dim3 blocksPerGrid(num_out_fm,1,1);
		    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);
			printf("threadsPerBlock for Conv1 = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	
			cudaEventRecord(start);
	
		    conv1_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_ifm, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
	    	
		    // d_ofm will now be used for the further layers 
		    err = cudaFree(d_ifm);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
		    err = cudaFree(d_mask);
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		    // Free host memory
		    free(h_ifm);
		    free(h_mask);
	
		}
		printf("conv1 done \n");
		// maxpooling 1 
		{
			// i/p : 96x55x55 , filter : 3x3 , stride 2 , o/p : 96x27x27
			int inp_r=55,  inp_c=55,  depth=96,  filter_width=3,  stride=2,  out_r=27,  out_c=27;
		    int numElements = inp_r*inp_c*depth;
		    int numElements_out = out_r*out_c*depth;
		    size_t size = numElements * sizeof(float);
		    size_t size_out = numElements_out * sizeof(float);
	
		    // Allocate the device output vector C
		    d_out = NULL;
		    err = cudaMalloc((void **)&d_out, size_out);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		    
        	 dim3 blocksPerGrid(depth,1,1);
		    dim3 threadsPerBlock((out_r*out_c - 1)/granularity + 1,1 , 1);
		    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
	
		    // d_ofm is the o/p from the layer, will be the i/p of this 
		    shared_pool<<<blocksPerGrid, threadsPerBlock>>>(d_ofm, d_out, inp_r, inp_c, depth, filter_width, stride, out_r, out_c, granularity);
		    err = cudaGetLastError();
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		    // Free device global memory , d_ofm is the o/p of the Conv1 , not needed amymore 
		    err = cudaFree(d_ofm);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		} // o/p of maxpooling is in d_out 
		printf("maxpooling 1 Done. \n");
		// Conv2
		{
			if(true){
	
				// i/p : 96x27x27, o/p : 256x27x27, filter : 5x5x256x48 , padding : 2 
				int num_in_fm = 48;
			   	int in_fm_h = 27;
			   	int in_fm_w = 27;
			   	int num_out_fm = 256;
			   	int out_fm_w = 27;
			   	int out_fm_h = 27;
			   	int mask_size = 5;
			   	int stride = 1;
			   	int pad = 2;
			   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
	
			   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
			   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
			   	float *h_mask = conv2.weights;
			   	printf(" In conv 2 \n");
				printf(" filter_size : %d , num_layers : %d, depth : %d \n",conv2.filter_size,conv2.num_layers,conv2.depth);
	
			   	
			    d_ofm = NULL;
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
	
				err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			
			    dim3 blocksPerGrid(num_out_fm,1,1);
			    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);
				printf("threadsPerBlock for Conv1 = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	
				cudaEventRecord(start);
	
			    conv2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_ofm, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
		    	
			    // d_ofm will now be used for the further layers 
			    err = cudaFree(d_out);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
			    err = cudaFree(d_mask);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			    // Free host memory
			    free(h_mask);
	
			} 
		}	// o/p is d_ofm 
		printf("Conv2 Done \n");
		// maxpooling 2
		{
			// i/p : 256x27x27 , filter : 3x3 , stride 2 , o/p : 265x13x13
			int inp_r=27,  inp_c=27,  depth=256,  filter_width=3,  stride=2,  out_r=13,  out_c=13;
		    int numElements = inp_r*inp_c*depth;
		    int numElements_out = out_r*out_c*depth;
		    size_t size = numElements * sizeof(float);
		    size_t size_out = numElements_out * sizeof(float);
	
		    // Allocate the device output vector C
		    d_out = NULL;
		    err = cudaMalloc((void **)&d_out, size_out);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
	     	int threadsPerBlock = (out_r*out_c - 1)/granularity + 1;
        	int blocksPerGrid = depth;
		    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	
		    // d_ofm is the o/p from the layer, will be the i/p of this 
		    shared_pool<<<blocksPerGrid, threadsPerBlock>>>(d_ofm, d_out, inp_r, inp_c, depth, filter_width, stride, out_r, out_c, granularity);
		    err = cudaGetLastError();
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		    // Free device global memory , d_ofm is the o/p of the Conv1 , not needed amymore 
		    err = cudaFree(d_ofm);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		} // o/p of maxpooling is in d_out 
		printf("maxpooling 2 \n");
		// Conv3
		{
			 
			if(true){
	
				// i/p : 256x13x13, o/p : 384x13x13, filter : 3x3x384x256 , padding : 1 
				int num_in_fm = 256;
			   	int in_fm_h = 13;
			   	int in_fm_w = 13;
			   	int num_out_fm = 384;
			   	int out_fm_w = 13;
			   	int out_fm_h = 13;
			   	int mask_size = 3;
			   	int stride = 1;
			   	int pad = 1;
			   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
	
			   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
			   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
			   	float *h_mask = conv3.weights;
				printf(" In conv 3 \n");
				printf(" filter_size : %d , num_layers : %d, depth : %d \n",conv3.filter_size,conv3.num_layers,conv3.depth);
	
			    d_ofm_3 = NULL;
			    err = cudaMalloc((void **)&d_ofm_3, out_size);
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
	
				err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			
			    dim3 blocksPerGrid(num_out_fm,1,1);
			    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);
				printf("threadsPerBlock for Conv1 = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	
				cudaEventRecord(start);
	
			    conv2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_ofm_3, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
		    	
			    // d_ofm will now be used for the further layers 
			    err = cudaFree(d_out);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
			    err = cudaFree(d_mask);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			    // Free host memory
			    free(h_mask);
	
			} 
		}	// o/p is d_ofm_3 
		printf("Conv3 Done\n");
		// Conv4
		{
			 
	
			if(true){
				ConvLayer c = conv4;
	
				// i/p : 384x13x13, o/p : 384x13x13, filter : 3x3x384x192 , padding : 1 
				int num_in_fm = 192;
			   	int in_fm_h = 13;
			   	int in_fm_w = 13;
			   	int num_out_fm = 384;
			   	int out_fm_w = 13;
			   	int out_fm_h = 13;
			   	int mask_size = 3;
			   	int stride = 1;
			   	int pad = 2;
			   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
	
			   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
			   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
			   	float *h_mask = c.weights;
				printf(" In conv 4 \n");
				printf(" filter_size : %d , num_layers : %d, depth : %d \n",conv4.filter_size,conv4.num_layers,conv4.depth);
		
			    d_ofm_4 = NULL;
			    err = cudaMalloc((void **)&d_ofm_4, out_size);
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
	
				err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			
			    dim3 blocksPerGrid(num_out_fm,1,1);
			    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);
				printf("threadsPerBlock for Conv1 = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	
				cudaEventRecord(start);
	
			    conv2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_ofm_3, d_ofm_4, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
		    	
			    // d_ofm will now be used for the further layers 
			    err = cudaFree(d_ofm_3);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
			    err = cudaFree(d_mask);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			    // Free host memory
			    free(h_mask);
	
			} 
		}	// o/p is d_ofm_4 
		printf("Conv4 Done\n");
		// Conv5
		{
			 
			getline(in, line);
			std::stringstream lineStream(line);
			if(line[0] == 'c'){
				ConvLayer c = processConv(lineStream);
	
				// i/p : 384x13x13, o/p : 256x13x13, filter : 3x3x256x192 , padding : 1 
				int num_in_fm = 192;
			   	int in_fm_h = 13;
			   	int in_fm_w = 13;
			   	int num_out_fm = 256;
			   	int out_fm_w = 13;
			   	int out_fm_h = 13;
			   	int mask_size = 3;
			   	int stride = 1;
			   	int pad = 2;
			   	int in_size = num_in_fm*in_fm_w*in_fm_h * sizeof(float);
	
			   	int out_size = num_out_fm*out_fm_w*out_fm_w * sizeof(float);
			   	int total_mask_size = num_out_fm*num_in_fm*mask_size*mask_size*sizeof(float);
			   	float *h_mask = conv5.weights;
			   	float *test_ofm = (float *) malloc(out_size);
				printf(" In conv 5 \n");
				printf(" filter_size : %d , num_layers : %d, depth : %d \n",conv5.filter_size,conv5.num_layers,conv5.depth);
	
			    d_ofm_5 = NULL;
			    err = cudaMalloc((void **)&d_ofm_5, out_size);
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
	
				err = cudaMemcpy(d_mask, h_mask, total_mask_size, cudaMemcpyHostToDevice);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to copy matrix mask from host to device (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			
			    dim3 blocksPerGrid(num_out_fm,1,1);
			    dim3 threadsPerBlock(out_fm_w, ((out_fm_h + granularity - 1)/granularity) , 1);
				printf("threadsPerBlock for Conv1 = %d,%d,%d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	
				cudaEventRecord(start);
	
			    conv2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_ofm_4, d_ofm_5, d_mask, in_fm_h, in_fm_w, num_in_fm, out_fm_h, out_fm_w, num_out_fm, mask_size, pad, stride, granularity);
		    	
			    // d_ofm will now be used for the further layers 
			    err = cudaFree(d_ofm_4);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix ifm (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
			    err = cudaFree(d_mask);
			    if (err != cudaSuccess)
			    {
			        fprintf(stderr, "Failed to free device matrix mask (error code %s)!\n", cudaGetErrorString(err));
			        exit(EXIT_FAILURE);
			    }
	
			    // Free host memory
			    free(h_mask);
	
			} 
		}	// o/p id d_ofm_5
		printf("Conv5 Done \n");
		// maxpooling 3
		{
			// i/p : 256x13x13 , filter : 3x3 , stride 2 , o/p : 265x6x6
			int inp_r=13,  inp_c=13,  depth=256,  filter_width=3,  stride=2,  out_r=6,  out_c=6;
		    int numElements = inp_r*inp_c*depth;
		    int numElements_out = out_r*out_c*depth;
		    size_t size = numElements * sizeof(float);
		    size_t size_out = numElements_out * sizeof(float);
	
		    // Allocate the device output vector C
		    d_out = NULL;
		    err = cudaMalloc((void **)&d_out, size_out);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
     		int threadsPerBlock = (out_r*out_c - 1)/granularity + 1;
        	int blocksPerGrid = depth;
		    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	
		    // d_ofm is the o/p from the layer, will be the i/p of this 
		    shared_pool<<<blocksPerGrid, threadsPerBlock>>>(d_ofm_5, d_out, inp_r, inp_c, depth, filter_width, stride, out_r, out_c, granularity);
		    err = cudaGetLastError();
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		    // Free device global memory , d_ofm is the o/p of the Conv1 , not needed amymore 
		    err = cudaFree(d_ofm);
	
		    if (err != cudaSuccess)
		    {
		        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		        exit(EXIT_FAILURE);
		    }
	
		} // o/p of maxpooling is in d_out 
		printf("maxpooling 3 done \n");
		// FC 6 
		{
			// if_vector : 256x6x6 matrix : 
			FCLayer f = fc6;
	    	int numARows = fc6.outputs;   // number of rows in the ifm 
			int numAColumns = fc6.inputs;  // number of columns in the ifm
			int numBRows = 256*6*6;   // number of rows in the vector
			int numBColumns=1;  // number of columns in the vector
			int numCRows = numARows;  // number of rows in the matrix C (you have to set this)
			int numCColumns=1; // number of columns in the matrix C (you have to set this)
			int nelem_per_thread = granularity; // THread coarsening factor
			float  *hostBias, *deviceBias, *matrix;
			cudaError_t err = cudaMalloc((void **)&out_1, sizeof(float)*numCRows*numCColumns);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	        hostBias = fc6.biases;
	        err = cudaMalloc((void **)&deviceBias, sizeof(float)*numCRows*numCColumns);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaMemcpy(deviceBias, hostBias, sizeof(float)*numCRows*numCColumns, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }

	        err = cudaMalloc((void **)&matrix, sizeof(float)*fc6.outputs*fc6.inputs);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaMemcpy(matrix, fc6.weights, sizeof(float)*f.outputs*f.inputs, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        // Initialize the grid and block dimensions
		    // Launch the Vector Add CUDA Kernel
		    int numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
		    int threadsPerBlock = 256;
		    int blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
		    dim3 dimGrid(blocksPerGrid, 1, 1);//Number of Blocks required
		    dim3 dimBlock(threadsPerBlock, 1, 1);//Number of threads in each block
			
		    // Shared memory for parameter vetor and bias values
		    int totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block
		    printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);
	
		    gen_matvec<<<dimGrid, dimBlock, totSharedMem>>>(matrix, d_out, out_1, deviceBias, numCRows, numAColumns, nelem_per_thread);
	
		    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call
	
		    err = cudaFree(matrix);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaFree(d_out);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
		}	 // out is the output 
		printf("FC6 done\n");
	
		// FC7 
		{
			// ip matrix : 4096x4096 , output vector : 4096x1 
			FCLayer f = fc7;
	    	int numARows = f.outputs;   // number of rows in the ifm 
			int numAColumns = f.inputs;  // number of columns in the ifm
			int numBRows = 4096;   // number of rows in the ofm
			int numBColumns=1;  // number of columns in the ofm
			int numCRows = numARows;  // number of rows in the matrix C (you have to set this)
			int numCColumns=1; // number of columns in the matrix C (you have to set this)
			int nelem_per_thread = granularity; // THread coarsening factor
			float  *hostBias, *deviceBias, *matrix;
			cudaError_t err = cudaMalloc((void **)&out_2, sizeof(float)*numCRows*numCColumns);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	        hostBias = f.biases;
	        err = cudaMalloc((void **)&deviceBias, sizeof(float)*numCRows*numCColumns);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaMemcpy(deviceBias, hostBias, sizeof(float)*numCRows*numCColumns, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }

	        err = cudaMalloc((void **)&matrix, sizeof(float)*f.outputs*f.inputs);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaMemcpy(matrix, f.weights, sizeof(float)*f.outputs*f.inputs, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        // Initialize the grid and block dimensions
		    // Launch the Vector Add CUDA Kernel
		    int numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
		    int threadsPerBlock = 256;
		    int blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
		    dim3 dimGrid(blocksPerGrid, 1, 1);//Number of Blocks required
		    dim3 dimBlock(threadsPerBlock, 1, 1);//Number of threads in each block
			
		    // Shared memory for parameter vetor and bias values
		    int totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block
		    printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);
	
		    gen_matvec<<<dimGrid, dimBlock, totSharedMem>>>(matrix, out_1, out_2, deviceBias, numCRows, numAColumns, nelem_per_thread);
	
		    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call
	
		    err = cudaFree(matrix);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaFree(out_1);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
		}	 // out_2 is the output 
		printf("FC7 done \n");
		// FC8
		{
			// ip matrix : 1000x4096 , output vector : 1000x1 
			FCLayer f = fc8;
	    	int numARows = f.outputs;   // number of rows in the ifm 
			int numAColumns = f.inputs;  // number of columns in the ifm
			int numBRows = 4096;   // number of rows in the vector
			int numBColumns=1;  // number of columns in the vector
			int numCRows = numARows;  // number of rows in the matrix C (you have to set this)
			int numCColumns=1; // number of columns in the matrix C (you have to set this)
			int nelem_per_thread = granularity; // THread coarsening factor
			float  *hostBias, *deviceBias, *matrix;
			cudaError_t err = cudaMalloc((void **)&out_3, sizeof(float)*numCRows*numCColumns);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	        hostBias = f.biases;
	        err = cudaMalloc((void **)&deviceBias, sizeof(float)*numCRows*numCColumns);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaMemcpy(deviceBias, hostBias, sizeof(float)*numCRows*numCColumns, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	

			err = cudaMalloc((void **)&matrix, sizeof(float)*f.outputs*f.inputs);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }

	        err = cudaMemcpy(matrix, f.weights, sizeof(float)*f.outputs*f.inputs, cudaMemcpyHostToDevice);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        // Initialize the grid and block dimensions
		    // Launch the Vector Add CUDA Kernel
		    int numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
		    int threadsPerBlock = 256;
		    int blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
		    dim3 dimGrid(blocksPerGrid, 1, 1);//Number of Blocks required
		    dim3 dimBlock(threadsPerBlock, 1, 1);//Number of threads in each block
			
		    // Shared memory for parameter vetor and bias values
		    int totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block
		    printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);
	
		    gen_matvec<<<dimGrid, dimBlock, totSharedMem>>>(matrix, out_2, out_3, deviceBias, numCRows, numAColumns, nelem_per_thread);
	
		    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call
	
		    err = cudaFree(matrix);
			if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
	        err = cudaFree(out_2);
	        if (err != cudaSuccess) {
	            printf( "Failed to run stmt %d ", __LINE__);
	            return -1;
	        }
	
		}	 // out_2 is the output 

		printf("FC8 Done \n");
		cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
		delta = 0;
		cudaEventElapsedTime(&delta, start, stop);
		printf("conv2, shared_pool, gen_matvec\n");
		printf("granularity = %d, time in milliseconds = %f\n",granularity,delta);
	}



	
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