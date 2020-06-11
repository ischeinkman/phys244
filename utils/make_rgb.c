#include "../cudac/mycomplex.h"
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include <syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#define KILL_ON_ERR(expr)                                                                     \
    do                                                                                        \
    {                                                                                         \
        int code = (expr);                                                                    \
        if (code < 0)                                                                         \
        {                                                                                     \
            int mmmerr = (code == -1) ? errno : code;                                         \
            fprintf(stderr, "Line %d: " #expr " returned error code %d\n", __LINE__, mmmerr); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)
#define KILL_ON_NULL(expr)                                                              \
    do                                                                                  \
    {                                                                                   \
        void *ptr = (expr);                                                             \
        if (ptr == NULL)                                                                \
        {                                                                               \
            fprintf(stderr, "Line %d: " #expr " returned a null pointer.\n", __LINE__); \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#ifndef WIDTH
#define WIDTH (100)
#endif
#ifndef HEIGHT
#define HEIGHT (100)
#endif
#ifndef NUM_FRAMES
#define NUM_FRAMES (500)
#endif

#define FRAME_SIZE (WIDTH * HEIGHT)
#define COMPLEX_SIZE 8
#define PIXEL_SIZE (3)
#define FRAME_BYTES (PIXEL_SIZE * FRAME_SIZE)
#define FRAME_CMPBT (FRAME_SIZE * COMPLEX_SIZE)

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        fprintf(stderr, "USAGE: %s <input file> <output file>\n", argv[0]);
        exit(1);
    }
    const long page_size = sysconf(_SC_PAGE_SIZE);
    char *input_file = argv[1];
    int input_fd;
    KILL_ON_ERR(input_fd = open(input_file, O_RDONLY));

    char *output_file = argv[2];
    int output_fd;
    KILL_ON_ERR(output_fd = open(output_file, O_RDWR | O_TRUNC | O_CREAT, 0x1b6));

    struct stat sb;
    KILL_ON_ERR(fstat(input_fd, &sb));

    uint8_t *inmap;
    MyComplex *buffer;
    KILL_ON_NULL(buffer = (MyComplex *)malloc(FRAME_SIZE * sizeof(MyComplex)));
    for (int cur_frame = 0; cur_frame < NUM_FRAMES; cur_frame++)
    {
        int complex_offset = cur_frame * FRAME_CMPBT;
        int pixel_offset = cur_frame * FRAME_BYTES;

        long inmap_offset = (complex_offset / page_size) * page_size;
        long inmap_diff = complex_offset - inmap_offset;
        KILL_ON_NULL(inmap = mmap(NULL, FRAME_CMPBT + inmap_diff, PROT_READ, MAP_PRIVATE, input_fd, inmap_offset));
        if (inmap == (void *)(-1))
        {
            KILL_ON_ERR(-1);
        }
        inmap += inmap_diff;

        long outmap_offset = (pixel_offset / page_size) * page_size;
        long outmap_diff = pixel_offset - outmap_offset;

        MyComplex_read(inmap, FRAME_CMPBT, buffer, FRAME_SIZE);
        float max_p = 0.0;
        float sum_p = 0.0;
        for (int idx = 0; idx < FRAME_SIZE; idx++)
        {
            MyComplex cur = buffer[idx];
            float p = MyComplex_mag_2(cur);
            sum_p += p;
            if (p > max_p)
            {
                max_p = p;
            }
        }
        if (abs(sum_p - 1.0) >= 0.0001)
        {
            fprintf(stderr, "   BAD P: %f\n", sum_p);
            exit(1);
        }
        float sqrt_maxp = sqrt(max_p);
        for (int idx = 0; idx < FRAME_SIZE; idx++)
        {
            MyComplex cur = buffer[idx];
            float cur_p = MyComplex_mag_2(cur);
            uint8_t r = (uint8_t)(roundf( 255.0 * fabsf(cur.re/sqrt_maxp) ));
            uint8_t g = (uint8_t)(roundf(255.0  * fabsf(cur.im/sqrt_maxp)));
            uint8_t b = 0;
            uint8_t colors[3] = {r, g, b};
            write(output_fd, colors, 3);
        }
        KILL_ON_ERR(munmap(inmap - inmap_diff, FRAME_SIZE * COMPLEX_SIZE + inmap_diff));
        KILL_ON_ERR(fsync(output_fd));
    }

    KILL_ON_ERR(close(input_fd));
    KILL_ON_ERR(close(output_fd));
    free(buffer);
}
