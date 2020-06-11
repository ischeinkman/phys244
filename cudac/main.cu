#include "mycomplex.h"
#include "stepper.cu"
#include "summer.cu"
#include "consts.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#ifndef HBAR
#define HBAR (1.0)
#endif

#ifndef M
#define M (1.0)
#endif

#ifndef DX
#define DX (0.1)
#endif

#ifndef DT
#define DT (0.1)
#endif

#ifndef WIDTH
#define WIDTH (500)
#endif

#ifndef HEIGHT
#define HEIGHT (500)
#endif

#ifndef NUM_FRAMES
#define NUM_FRAMES (500)
#endif

#ifndef SLIT_HEIGHT
#define SLIT_HEIGHT (-100)
#endif

#ifndef SLIT_1_START
#define SLIT_1_START (1)
#endif 

#ifndef SLIT_1_END
#define SLIT_1_END (1)
#endif 

#ifndef SLIT_2_START
#define SLIT_2_START (1)
#endif 

#ifndef SLIT_2_END
#define SLIT_2_END (1)
#endif 


#define FRAME_SIZE (WIDTH * HEIGHT)
#define FRAME_BYTES (FRAME_SIZE * 8)
#define TOTAL_BUFFER_SIZE (NUM_FRAMES * FRAME_SIZE)
#define TOTAL_BUFFER_BYTES (TOTAL_BUFFER_SIZE * 8)

#define STABILITY_COEFF (1.0 - 2.0 * DT * HBAR / (M * DX * DX))
#define IS_STABLE (STABILITY_COEFF > 0.0)

#define OFFSET_FOR(xidx, yidx) (xidx + (yidx * WIDTH))

#define CLOCK_TO_MILLIS(clk) ( (long) (( ( (float) clk) * 1000.0) / CLOCKS_PER_SEC) )

#ifndef on_border
#define on_border(xidx, yidx) ((xidx == 0 || yidx == 0 || xidx >= WIDTH - 1 || yidx >= HEIGHT - 1))
#endif 

#ifndef in_slit 
#define in_slit(x, y) (                        \
    (y == SLIT_HEIGHT) &&                      \
    ((SLIT_1_START <= x && x <= SLIT_1_END) || \
     (SLIT_2_START <= x && x <= SLIT_2_END)))
#endif 

#ifndef NUM_WORKGROUPS
#define NUM_WORKGROUPS (DISPATCH_X * DISPATCH_Y)
#endif
#ifndef RELEVANT_WORKGROUPS
#define RELEVANT_WORKGROUPS (WORKGROUP_WIDTH * WORKGROUP_HEIGHT)
#endif

float sum_p(MyComplex *frame)
{
    float retvl = 0.0;
    for (int idx = 0; idx < FRAME_SIZE; idx++)
    {
        MyComplex current = frame[idx];
        retvl += (current.re * current.re + current.im * current.im);
    }
    return retvl;
}

void normalize(MyComplex *frame)
{
    float psum = sum_p(frame);
    float coeff = 1.0 / sqrtf(psum);
    for (int idx = 0; idx < FRAME_SIZE; idx++)
    {
        frame[idx].re *= coeff;
        frame[idx].im *= coeff;
    }
}
void initialize(MyComplex *frame)
{
    for (int yidx = 0; yidx < HEIGHT; yidx += 1)
    {
        for (int xidx = 0; xidx < WIDTH; xidx += 1)
        {
            int cur_idx = OFFSET_FOR(xidx, yidx);
            if (xidx == WIDTH / 2 && yidx == HEIGHT / 2)
            {
                frame[cur_idx] = (MyComplex){.re = 1.0, .im = 0.0};
            }
            else if ((WIDTH / 2 - 1 <= xidx && xidx <= WIDTH / 2 + 1) && (HEIGHT / 2 - 1 <= yidx && yidx <= HEIGHT / 2 + 1))
            {
                frame[cur_idx] = MyComplex_im(0.5);
            }
            else
            {
                frame[cur_idx] = MyComplex_zero();
            }
        }
    }
    normalize(frame);
}

float * createPbuffer(float initial_p) {
    float * tmp = (float * ) calloc(NUM_WORKGROUPS + 1, sizeof(float));
    tmp[RELEVANT_WORKGROUPS] = initial_p;
    float * retvl;
    checkCudaErrors(cudaMalloc(&retvl, (NUM_WORKGROUPS + 1) * sizeof(float) ));
    checkCudaErrors(cudaMemcpy(retvl, tmp, (NUM_WORKGROUPS + 1) * sizeof(float), cudaMemcpyHostToDevice ));
    free(tmp);
    return retvl;
}

int main(int argc, const char **argv)
{
    MyComplex *cpu_alloc = (MyComplex *)calloc(FRAME_SIZE * NUM_FRAMES, sizeof(MyComplex));
    initialize(cpu_alloc);
    float initial_p = sum_p(cpu_alloc);
    fprintf(stderr, "Made CPU initial frame \n");
    clock_t start = clock();
    
    int dev = findCudaDevice(argc, argv);

    float2 * raw_allocation;
    checkCudaErrors(cudaMalloc(&raw_allocation, FRAME_SIZE * NUM_FRAMES * sizeof(float2)));
    checkCudaErrors(cudaMemcpy( raw_allocation, cpu_alloc, FRAME_SIZE * sizeof(MyComplex), cudaMemcpyHostToDevice ));
    

    float * pbuffer = createPbuffer(initial_p);


    dim3 threads(LAYOUT_X, LAYOUT_Y);
    dim3 grid(DISPATCH_X, DISPATCH_Y);

    for(uint idx = 0; idx < NUM_FRAMES-1; idx++) {
        stepper<<< grid, threads >>>(raw_allocation, pbuffer, idx);
        checkCudaErrors( cudaDeviceSynchronize() );
        summer<<< grid, threads >>>(pbuffer);
        checkCudaErrors( cudaDeviceSynchronize() );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors(cudaMemcpy(  cpu_alloc, raw_allocation, NUM_FRAMES * FRAME_SIZE * sizeof(MyComplex), cudaMemcpyDeviceToHost ));

    clock_t end = clock();
    clock_t diff = end - start; 
    printf("%d / %d\n", CLOCK_TO_MILLIS(diff), CLOCK_TO_MILLIS(diff)/NUM_FRAMES);

    FILE * outfile = fopen("cuda_out.data", "w");
    fwrite(cpu_alloc, sizeof(MyComplex), NUM_FRAMES * FRAME_SIZE, outfile);
    fclose(outfile);
    cudaFree(raw_allocation);
    free(cpu_alloc);
    return 0;
}
