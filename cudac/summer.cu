
#define WORKGROUP_SIZE (LAYOUT_X * LAYOUT_Y)

#define WORKGROUP_WIDTH (uint(WIDTH % LAYOUT_X  != 0) + WIDTH/LAYOUT_X)
#define WORKGROUP_HEIGHT (uint(HEIGHT % LAYOUT_Y  != 0) + HEIGHT/LAYOUT_Y)
#define NUM_POINTS (WORKGROUP_WIDTH * WORKGROUP_HEIGHT)


//layout(local_size_x = LAYOUT_X * LAYOUT_Y, local_size_y = 1, local_size_z = 1) in;

__global__ void summer(float * p_sums) {
    uint my_idx = threadIdx.x;
    for(uint current_batch = NUM_POINTS; current_batch > 1 ; current_batch /=2) {
        if(my_idx < current_batch/2) {
            p_sums.data[my_idx] += p_sums.data[current_batch/2 + my_idx];
        }
        __syncthreads();
    }
    p_sums[NUM_POINTS] = p_sums[0];
}