#include "gen_gpu.h"


//Function To handle any errors occurred in the function calls
int funcCheck(cudaError_t stmt){
     do {
            cudaError_t err = stmt;
            if (err != cudaSuccess) {
                printf( "Failed to run stmt %d ", __LINE__);
                return -1;
            }
        } while(0);
    return 0;
}

int numARows;   // number of rows in the matrix A
int numAColumns;  // number of columns in the matrix A
int numBRows;   // number of rows in the matrix B
int numBColumns=1;  // number of columns in the matrix B
int numCRows;  // number of rows in the matrix C (you have to set this)
int numCColumns=1; // number of columns in the matrix C (you have to set this)
int nelem_per_thread; // THread coarsening factor

//*************************************************************
void Print_Mat(int Row,int Col,float *Mat)//Function To print the Matrix
{
 int tot = Row*Col;
 for(int i=0;i<tot;i++)
   {
   printf("%f  ",Mat[i]);

   if((i%Col)==0 )
    {
     printf("\n");
    }
   }
}

//Function close
//*************************************************************
//Normal CPU Matrix Multiplication
void matMultiplyAddOnHost(float * A, float * B, float * C, float *bias, int numARows,
                        int numAColumns, int numBRows, int numBColumns,
                        int numCRows, int numCColumns)
{
    for (int i=0; i < numARows; i ++)
    {
        for (int j = 0; j < numBColumns; j++)
        {
            float c = 0.0;
            for (int k = 0; k < numAColumns; k++)
            {
                c += A[i*numAColumns + k] * B[k*numBColumns + j];
            }
        C[i*numCColumns + j] = c + bias[i*numCColumns + j];
        }
    }
}
//*************************************************************
int main(int argc, char ** argv) {
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * hostBias; // The Bias Matrix
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;
    float * deviceBias;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float delta = 0.0; //to measure time
    
    // //fc7 dimensions:
    // numARows = 4096;
    // numAColumns = 256*6*6;
    // numBRows = numAColumns;
    // nelem_per_thread = 4;

    // //fc8 dimensions:
    // numARows = 4096;
    // numAColumns = 4096;
    // numBRows = numAColumns;
    // nelem_per_thread = 4;	

    //fc9 dimensions:
    numARows = 4096;
    numAColumns = 1000;
    numBRows = numAColumns;
    nelem_per_thread = 4;

    // printf("\nPlease Enter Rows of B:");
    // scanf("%d %d",&numBRows);

    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    for (int i = 0; i < numARows*numAColumns; i++)//Matrix Initialization
    {
        hostA[i]=1.0;
    }
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostB[i]=1.0;
    }

    // Setting numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostBias = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    for (int i = 0; i < numCRows*numCColumns; i++)
    {
        hostBias[i]=1.0;
    }

    // Allocating GPU memory
    funcCheck(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
    funcCheck(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns));
    funcCheck(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));
    funcCheck(cudaMalloc((void **)&deviceBias, sizeof(float)*numCRows*numCColumns));

    // Copy memory to the GPU
    funcCheck(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(deviceBias, hostBias, sizeof(float)*numCRows*numCColumns, cudaMemcpyHostToDevice));

    // Initialize the grid and block dimensions
    // Launch the Vector Add CUDA Kernel
    nelem_per_thread=1;
    int numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
    int threadsPerBlock = (256+nelem_per_thread-1)/nelem_per_thread;
    int blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dimGrid(blocksPerGrid, 1, 1);//Number of Blocks required
    dim3 dimBlock(threadsPerBlock, 1, 1);//Number of threads in each block
    
    // Shared memory for parameter vetor and bias values
    int totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block
    //int totSharedMem = (threadsPerBlock * nelem_per_thread * numAColumns + numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block
    gen_matvec<<<dimGrid, dimBlock, totSharedMem>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns, nelem_per_thread);
    // float *A, float *x, float *y, const int m, const int n
    //@@ Launch the GPU Kernel here

    double avgtime[33];
    memset(avgtime,0.0,sizeof(double)*33);
    double avgtime_shared[33];
    memset(avgtime_shared,0.0,sizeof(double)*33);
    int nruns = 30;
    for(int runs=0;runs<nruns;++runs){
	    for(nelem_per_thread = 1; nelem_per_thread <= 32; nelem_per_thread++ ) {

	    	// Initialize the grid and block dimensions
		    // Launch the Vector Add CUDA Kernel
		    numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
		    threadsPerBlock = (256+nelem_per_thread-1)/nelem_per_thread;
		    blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
		    dim3 dimGrid_tmp(blocksPerGrid, 1, 1);//Number of Blocks required
		    dim3 dimBlock_tmp(threadsPerBlock, 1, 1);//Number of threads in each block
		    
		    // Shared memory for parameter vetor and bias values
		    totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block

		    printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);

		    cudaEventRecord(start);

		    gen_matvec<<<dimGrid_tmp, dimBlock_tmp, totSharedMem>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns, nelem_per_thread);
		    cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            delta = 0;
            cudaEventElapsedTime(&delta, start, stop);
            double time_taken = delta;
		    avgtime[nelem_per_thread]+=time_taken;
		    printf("Time taken is - %lf milliseconds\n ", time_taken);

		    cudaEventRecord(start);
		    gen_matvec_noshared<<<dimGrid_tmp, dimBlock_tmp>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns, nelem_per_thread);
		    cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            delta = 0;
            cudaEventElapsedTime(&delta, start, stop);
            time_taken = delta;
		    avgtime_shared[nelem_per_thread]+=time_taken;
		    printf("Time taken is - %lf milliseconds\n ", time_taken);
		    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call
	    }
	}
	printf("\nshared-memory\n:");
	for(nelem_per_thread = 1; nelem_per_thread <= 32; nelem_per_thread++ ){
		double time = avgtime[nelem_per_thread]/nruns;
		numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
	    threadsPerBlock = (256+nelem_per_thread-1)/nelem_per_thread;
	    blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
	    
	    // Shared memory for parameter vetor and bias values
	    totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block

	    // printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);
	    printf("%lf, ", time);
	}
	printf("\nGlobal-memory\n:");
	for(nelem_per_thread = 1; nelem_per_thread <= 32; nelem_per_thread++ ){
		double time = avgtime_shared[nelem_per_thread]/nruns;
		numThreadsReq = (numCRows+nelem_per_thread-1)/nelem_per_thread;
	    threadsPerBlock = (256+nelem_per_thread-1)/nelem_per_thread;
	    blocksPerGrid =(numThreadsReq + threadsPerBlock - 1) / threadsPerBlock;
	    
	    // Shared memory for parameter vetor and bias values
	    totSharedMem = (numAColumns + numCRows*numCColumns)* sizeof(float); // Shared memory per block

	    // printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);
	    printf("%lf, ", time);
	}
    printf("\n");
    //cudaDeviceSynchronize();//To synchronize the device

    // Copy the results in GPU memory back to the CPU
    funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

    matMultiplyAddOnHost(hostA, hostB, hostComputedC, hostBias, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    for (int i=0; i < numCColumns*numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
    {
        if (hostComputedC[i]  != hostC[i] )
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }

    // printf("\n Number of Blocks Created:%d \n",((numCColumns/Tile_size) + 1)*((numCColumns/Tile_size) + 1));
    // printf("\n Number of Threads Per Block: %d \n",(Tile_size*Tile_size));

    // Free the GPU memory
    funcCheck(cudaFree(deviceA));
    funcCheck(cudaFree(deviceB));
    funcCheck(cudaFree(deviceC));
    funcCheck(cudaFree(deviceBias));

    //Free the Pointer Memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostBias);
    free(hostComputedC);

    return 0;
}

