#include <stdio.h>
#define MAX_SZ 56
/*

//96*55*55 flattened vector -> inp(1D)  
//out size will depend on inp and filter size, stride.

//Keep calm, this code will be cleaned later!!
*/

__device__ int get_idx(int r, int c, int d, int rows, int cols) {
    return d*rows*cols + r*cols + c;
}

/*
//without granularity
__global__ void pooling(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c) {
    
    int gdimx, bid, bdimx, tid, id, r, c, d, rem, ir, ic, idx, i, j;
    float ans;
    gdimx = gridDim.x;
    bid = blockIdx.x; //1D Grid
    //bid = blockIdx.z * (gridDim.x *gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x; //3D Grid
    bdimx = blockDim.x;
    tid = threadIdx.x; //1D Block
    //tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x; //3D Block
    id = bid*bdimx + tid;
    if(id >= out_r*out_c*depth) {
        return;
    }
    //calculating position in output layer
    d = id/(inp_r*inp_c); //depth
    rem = id%(inp_r*inp_c);
    r = rem/inp_c; //row
    c = rem%inp_c; //col

    ir = stride*out_r; //input row
    ic = stride*out_c; //input col

    ans = INT_MIN;

    for(i=ir; i<ir+filter_width; i++) {
        idx = get_idx(i, j, d, inp_r, inp_c);
        for(j=ic; j<ic+filter_width; j++) {
            ans = max(ans, inp[idx++] );
        }
    }

    out[id] = ans;    
}
*/

//rows and cols are same.. redundant info can be skipped
__global__ void gran_pooling(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity) {
    int bid, bdimx, tid, init_id, id, r, c, d, rem, ir, ic, i, j;
    float ans;
    bid = blockIdx.x; //1D Grid
    //bid = blockIdx.z * (gridDim.x *gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x; //3D Grid
    bdimx = blockDim.x;
    tid = threadIdx.x; //1D Block
    //tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x; //3D Block
    init_id = bid*bdimx + tid;

    for(id = init_id*granularity; id < (init_id+1)*granularity; id++) {
        if(id >= out_r*out_c*depth) {
            return;
        }
        //calculating position in output layer
        d = id/(out_r*out_c); //depth
        rem = id%(out_r*out_c);
        r = rem/out_c; //row
        c = rem%out_c; //col

        ir = stride*r; //input row
        ic = stride*c; //input col

        ans = INT_MIN;

        for(i=ir; i<ir+filter_width; i++) {
            for(j=ic; j<ic+filter_width; j++) {
                ans = max(ans, inp[get_idx(i, j, d, inp_r, inp_c)] );
            }
        }

        out[id] = ans; 
        //printf("%d : r = %d, c = %d, d = %d, ans = %f\n", id, r, c, d, ans);
    }
}

//rows and cols are same.. redundant info can be skipped
__global__ void shared_pool(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity) {
    int bid, bdim, tid, id, r, c, d, ir, ic, i, j, skip_cells;
    float ans;
    bid = blockIdx.x; //1D Grid
    bdim = blockDim.x; 
    tid = threadIdx.x;
    d = bid;
    skip_cells = d*(inp_r*inp_c);
    
    //load data in shared memory
    const int size = inp_r*inp_c;
    __shared__ float shmem[MAX_SZ*MAX_SZ];
    int nload = (inp_r*inp_c - 1)/(bdim) + 1;
    
    for(i=nload*tid; i < (nload+1)*tid && i < size; i++) {
        shmem[i] = inp[skip_cells+i];
    }
    //loading done
    __syncthreads();
    
    
    for(id = tid*granularity; id < (tid+1)*granularity && id < out_r*out_c; id++) {
        //calculating position in output layer
        r = id/out_c; //row
        c = id%out_c; //col

        ir = stride*r; //input row
        ic = stride*c; //input col

        ans = INT_MIN;

        for(i=ir; i<ir+filter_width; i++) {
            for(j=ic; j<ic+filter_width; j++) {
                ans = max(ans, shmem[i*inp_c + j] );
            }
        }

        out[id] = ans; 
        //printf("%d : r = %d, c = %d, d = %d, ans = %f\n", skip_cells+id, r, c, d, ans);
    }
}

