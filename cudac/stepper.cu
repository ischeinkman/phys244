#include <cuComplex.h>

// For the sake of documentation, we add default values to all
// DEFINE-based constants wrapped inside ifndef blocks. 
// This allows for the constants to not be "magic" while still
// allowing for recompilation with different values. 
#ifndef HBAR
    #define HBAR 1
#endif

#ifndef M 
    #define M 1
#endif 

#ifndef DX
    #define DX 1
#endif 

#ifndef DT
    #define DT 1
#endif 

#ifndef WIDTH
    #define WIDTH 1
#endif 
#ifndef HEIGHT
    #define HEIGHT 1
#endif 
#ifndef SLIT_HEIGHT
    #define SLIT_HEIGHT 1
#endif 
#ifndef SLIT_1_START
    #define SLIT_1_START 1
#endif 
#ifndef SLIT_1_END
    #define SLIT_1_END 1
#endif 
#ifndef SLIT_2_START
    #define SLIT_2_START 1
#endif 
#ifndef SLIT_2_END
    #define SLIT_2_END 1
#endif 

#ifndef LAYOUT_X
    #define LAYOUT_X (1)
#endif

#ifndef LAYOUT_Y
    #define LAYOUT_Y (1)
#endif

// Macros to detect if this point needs to be computed 
#define on_border(x, y) (x <= 0 || y <= 0 || (x) >= WIDTH -1 || (y) >= HEIGHT -1 )
#define in_slit(x, y) ( (y) == SLIT_HEIGHT && (( (x) >= SLIT_1_START && x <= SLIT_2_END)  || ( (x) >= SLIT_2_START && x <= SLIT_2_END)))
#define OFFSET_TO(x, y) ( (x) + (y) * WIDTH)


#define oob(x, y) (x >= WIDTH || y >= HEIGHT)

// Coefficients of our finite-difference stepper
#define CUR_COEFF ( \
    make_cuFloatComplex ( \
        1.0,  \
        -2.0 * HBAR * DT/(M * DX * DX) \
    ) \
)
#define NEIGHBOR_COEFF (make_cuFloatComplex (0.0,  (HBAR * DT)/(2.0 * M * DX * DX)))

#define COMPLEX_MUL(a, b) ( cuCmulf(a, b) )

#define WORKGROUP_SIZE (LAYOUT_X * LAYOUT_Y)
#define WORKGROUP_WIDTH (uint(WIDTH % LAYOUT_X  != 0) + WIDTH/LAYOUT_X)
#define WORKGROUP_HEIGHT (uint(HEIGHT % LAYOUT_Y  != 0) + HEIGHT/LAYOUT_Y)
#define RELEVANT_WORKGROUPS (WORKGROUP_WIDTH * WORKGROUP_HEIGHT)
#define NUM_WORKGROUPS (DISPATCH_X * DISPATCH_Y)


//layout(local_size_x = LAYOUT_X, local_size_y = LAYOUT_Y, local_size_z = 1) in;


// CUDA's thread is equivalent to GLSL/Vulkan 's invocation, 
// and CUDA's block is equivalent to GLSL/Vulkan 's workgroup.


__global__ void stepper(float2 * input_data, float2 * output_data, float2 * p_sums) {
    __shared__ float WORKGROUP_PARTIALS[WORKGROUP_SIZE];
    uint x_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    uint y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    uint local_idx = threadIdx.y * blockDim.x + threadIdx.x; 
    float coeff = inversesqrt(p_sums[RELEVANT_WORKGROUPS]);

    float2 retvl = make_cuFloatComplex(-0.0, 0.0);
    float2 cur_value = input_data[OFFSET_TO(x_idx, y_idx)];
    if(!(on_border(x_idx, y_idx) || in_slit(x_idx, y_idx))) {
        float2 left_value = input_data[OFFSET_TO( (x_idx -1), y_idx)];
        float2 right_value = input_data[OFFSET_TO( (x_idx +1), y_idx)];
        float2 bottom_value = input_data[OFFSET_TO(x_idx , (y_idx -1))];
        float2 top_value = input_data[OFFSET_TO(x_idx, (y_idx +1))];

        float2 nsum = make_cuFloatComplex(
            left_value.x + right_value.x + top_value.x + bottom_value.x, 
            left_value.y + right_value.y + top_value.y + bottom_value.y 
        );

        float2 term_a = COMPLEX_MUL(CUR_COEFF, cur_value);
        float2 term_b = COMPLEX_MUL(NEIGHBOR_COEFF, nsum);
        retvl = make_cuFloatComplex(
            coeff * (term_a.x + term_b.x),
            coeff * (term_a.y + term_b.y)
        );
        output_data[OFFSET_TO(x_idx, y_idx)] = retvl;
    }
    float cur_p = dot(retvl, retvl) ;
    WORKGROUP_PARTIALS[local_idx] = max(cur_p, 0.0);
    __syncthreads();
    uint wk_x = blockIdx.x;
    uint wk_y = blockIdx.y;
    uint wk_width = WORKGROUP_WIDTH;
    uint my_offset = wk_x + wk_y * wk_width;
    float msum = 0.0;
    #pragma unroll
    for(uint k = 0; k < WORKGROUP_SIZE; k++) {
        uint other_idx = (local_idx + k) % WORKGROUP_SIZE;
        float found = WORKGROUP_PARTIALS[other_idx];
        msum += found;
    }
    if(my_offset < RELEVANT_WORKGROUPS) p_sums[my_offset] = msum;
}