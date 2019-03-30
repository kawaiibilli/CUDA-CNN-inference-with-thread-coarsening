#include <stdio.h>

/*

//96*55*55 flattened vector -> inp(1D)
//out size will depend on inp and filter size, stride.

*/

__device__ int get_idx(int r, int c, int d, int rows, int cols) {
    return d*rows*cols + r*cols + c;
}

//rows and cols are same.. redundant info can be skipped
__global__ void gran_pooling(float *inp, float *out, int inp_r, int inp_c, int depth, int filter_width, int stride, int out_r, int out_c, int granularity) {
    int gdimx, bid, bdimx, tid, init_id, id, r, c, d, rem, ir, ic, idx, i, j;
    float ans;
    bid = blockIdx.x; //1D Grid
    bdimx = blockDim.x;
    tid = threadIdx.x; //1D Block
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
        printf("%d : r = %d, c = %d, d = %d, ans = %f\n", id, r, c, d, ans);
    }
}