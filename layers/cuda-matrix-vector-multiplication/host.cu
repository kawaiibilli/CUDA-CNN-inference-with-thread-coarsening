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

 // number of rows in the matrix A
int numARows;  
// number of columns in the matrix A
int numAColumns;  
// number of rows in the vector B
int numBRows;   
// number of columns in the vector B
int numBColumns=1;  
// number of rows in the vector C 
int numCRows;  
// number of columns in the vector C 
int numCColumns=1; 
// THread coarsening factor
int nelem_per_thread; 




//Function To print the Matrix
void Print_Mat(int Row,int Col,float *Mat)
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


//Normal CPU Matrix Vector Multiplication
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



int main(int argc, char ** argv) {

    ////////////////////////////////////////////////INITIALIZING HOST AND DEVICE MEMORIES/////////////////////////////////////////
    float * hostA; // The A matrix
    float * hostB; // The B vector
    float * hostC; // The output C vector
    float * hostBias; // The Bias Matrix
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;
    float * deviceBias;
    /////////////////////////////////////////////////// C = A B + Bias ///////////////////////////////////////////////////////

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //to measure time
    float delta = 0.0; 
    
    // //fc7 dimensions:
    // numARows = 4096;
    // numAColumns = 256*6*6;
    // numBRows = numAColumns;
    // nelem_per_thread = 1;

    // //fc8 dimensions:
    // numARows = 4096;
    // numAColumns = 4096;
    // numBRows = numAColumns;
    // nelem_per_thread = 1;	

    //fc9 dimensions:
    numARows = 4096;
    numAColumns = 1000;
    numBRows = numAColumns;
    nelem_per_thread = 1;

    

    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    for (int i = 0; i < numARows*numAColumns; i++)
    {
        hostA[i]=rand()/(float) RAND_MAX;
    }
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostB[i]=rand()/(float) RAND_MAX;
    }

    // Setting numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostBias = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    for (int i = 0; i < numCRows*numCColumns; i++)
    {
        hostBias[i]=rand()/(float) RAND_MAX;
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
    //Number of Blocks required
    dim3 dimGrid(blocksPerGrid, 1, 1);
    //Number of threads in each block
    dim3 dimBlock(threadsPerBlock, 1, 1);
    
    // Shared memory for parameter vetor and bias values
    int totSharedMem = (numAColumns)* sizeof(float);

    //-----------------------------------------------------------------------------------------------------------------------------
    //////////////////////////////////////////TESTING NON THREAD COARSENING CODES///////////////////////////////////////////////////////////////
    
    //WITH SHARED MEMORY
    gen_matvec_nocoarse<<<dimGrid, dimBlock, totSharedMem>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns);

    //VERIFYING RESULT
    // Copy the results in GPU memory back to the CPU
    funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

    matMultiplyAddOnHost(hostA, hostB, hostComputedC, hostBias, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    for (int i=0; i < numCColumns*numCRows; i++)
    {
        if (abs(hostComputedC[i] - hostC[i]) > 1e-4 )
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            exit(1);
            
        }
    }


    ////WITHOUT SHARED MEMORY
    gen_matvec_nocoarse_noshared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns);

    //VERIFYING THE RESULT
    // Copy the results in GPU memory back to the CPU
    funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));
    matMultiplyAddOnHost(hostA, hostB, hostComputedC, hostBias, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    for (int i=0; i < numCColumns*numCRows; i++)
    {
        if (abs(hostComputedC[i] - hostC[i]) > 1e-4 )
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            exit(1);
            
        }
    }

    //-------------------------------------------------------------------------------------------------------------------------------
    //////////////////////////////////////////////// TESTING THREAD COARSENING CODES /////////////////////////////////////////////////


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
            //Number of Blocks required
		    dim3 dimGrid_tmp(blocksPerGrid, 1, 1);
            //Number of threads in each block
		    dim3 dimBlock_tmp(threadsPerBlock, 1, 1);
		    
		    // Shared memory for parameter vetor and bias values
		    totSharedMem = (numAColumns)* sizeof(float); // Shared memory per block

		    printf("CUDA kernel launch with %d blocks of %d threads, and %d of shared Memory\n", blocksPerGrid, threadsPerBlock, totSharedMem);

            //WITH SHARED MEMORY
		    cudaEventRecord(start);
		    gen_matvec<<<dimGrid_tmp, dimBlock_tmp, totSharedMem>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns, nelem_per_thread);
		    cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            delta = 0;
            cudaEventElapsedTime(&delta, start, stop);
            double time_taken = delta;
		    avgtime[nelem_per_thread]+=time_taken;
		    printf("Time taken is - %lf milliseconds\n ", time_taken);

            //VERIFYING THE RESULT
            // Copy the results in GPU memory back to the CPU
            funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));
            matMultiplyAddOnHost(hostA, hostB, hostComputedC, hostBias, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
            for (int i=0; i < numCColumns*numCRows; i++)
            {
                if (abs(hostComputedC[i] - hostC[i]) > 1e-4 )
                {
                    printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
                    exit(1);
                    
                }
            }

            //WITHOUT SHARED MEMORY
		    cudaEventRecord(start);
		    gen_matvec_noshared<<<dimGrid_tmp, dimBlock_tmp>>>(deviceA, deviceB, deviceC, deviceBias, numCRows, numAColumns, nelem_per_thread);
		    cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            delta = 0;
            cudaEventElapsedTime(&delta, start, stop);
            time_taken = delta;
		    avgtime_shared[nelem_per_thread]+=time_taken;
		    printf("Time taken is - %lf milliseconds\n ", time_taken);

            //VERIFYING THE RESULT
            // Copy the results in GPU memory back to the CPU
            funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));
            matMultiplyAddOnHost(hostA, hostB, hostComputedC, hostBias, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
            for (int i=0; i < numCColumns*numCRows; i++)
            {
                if (abs(hostComputedC[i] - hostC[i]) > 1e-4 )
                {
                    printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
                    exit(1);
                    
                }
            }


	    }

	}
    //---------------------------------------------------------------------------------------------------------------------------
    //////////////////////////////////////////////// PRINTING AND FREEING MEMORY//////////////////////////////////////////

	printf("\nshared-memory\n:");
	for(nelem_per_thread = 1; nelem_per_thread <= 32; nelem_per_thread++ ){
        double time = avgtime_shared[nelem_per_thread]/nruns;
		printf("%lf, ", time);
	}

	printf("\nGlobal-memory\n:");
	for(nelem_per_thread = 1; nelem_per_thread <= 32; nelem_per_thread++ ){
        double time = avgtime[nelem_per_thread]/nruns;
		printf("%lf, ", time);
	}
    printf("\n");

    


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

