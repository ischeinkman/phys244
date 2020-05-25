#include "mycomplex.h"
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

#ifndef FRAMES
#define FRAMES (500)
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

#define CLOCK_TO_MILLIS(clk) ((clk * 1000) / CLOCKS_PER_SECOND)
/*
#[inline(always)]
fn on_border(xidx: usize, yidx: usize) -> bool {
    xidx == 0 || yidx == 0 || xidx >= WIDTH - 1 || yidx >= HEIGHT - 1
}

#[inline(always)]
fn in_slit(x: usize, y: usize) -> bool {
    let x = x as isize;
    let y = y as isize;
    y == SLIT_HEIGHT
        && ((x >= SLIT_1_START && x <= SLIT_2_END) || (x >= SLIT_2_START && x <= SLIT_2_END))
}
*/
#define on_border(xidx, yidx) ((xidx == 0 || yidx == 0 || xidx >= WIDTH - 1 || yidx >= HEIGHT - 1))
#define in_slit(x, y) (                        \
    (y == SLIT_HEIGHT) &&                      \
    ((SLIT_1_START <= x && x <= SLIT_1_END) || \
     (SLIT_2_START <= x && x <= SLIT_2_END)))

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
}

int main(int argc, char **argv)
{
    printf("%s %s\n", IS_LE ? "T" : "F", IS_REAL_FIRST ? "T" : "F");
    MyComplex *allocation = (MyComplex *)calloc(FRAME_SIZE, sizeof(MyComplex));
    initialize(allocation);
    normalize(allocation);
    clock_t start = clock();
    return 0;
}