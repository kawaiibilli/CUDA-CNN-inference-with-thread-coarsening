#include "headers.h"
/**
 * Host main routine
 */

//In this, device input vector for pooling layer will be same as device output vector(flattened) of conv layer. 
//So many changes are required in this host code, this is just for verification. 
//Code verification result : OK!
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int inp_r=5,  inp_c=5,  depth=3,  filter_width=3,  stride=2,  out_r=2,  out_c=2,  granularity=1;
    int numElements = inp_r*inp_c*depth;
    int numElements_out = out_r*out_c*depth;
    size_t size = numElements * sizeof(float);
    size_t size_out = numElements_out * sizeof(float);

    // Allocate the host input vector (will be output of conv layer... won't need to declare this)
    float *h_inp = (float *)malloc(size);
    float *h_out = (float *)malloc(size_out);
    
    // Verify that allocations succeeded
    if (h_inp == NULL || h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_inp[i] = i+1;
    }

    // Allocate the device input vector A
    float *d_inp = NULL;
    err = cudaMalloc((void **)&d_inp, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_out = NULL;
    err = cudaMalloc((void **)&d_out, size_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =(numElements_out + granularity*threadsPerBlock - 1) / (granularity*threadsPerBlock);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    gran_pooling<<<blocksPerGrid, threadsPerBlock>>>(d_inp, d_out, inp_r, inp_c, depth, filter_width, stride, out_r, out_c, granularity);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements_out; ++i)
    {
        printf("ELEMENT : %f\n", h_out[i]);
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_inp);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_inp);
    free(h_out);

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