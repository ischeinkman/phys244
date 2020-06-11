#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
// This macro checks for endianness. 
// It works by setting up a union of a 32 bit number and 4 byte array, setting the number to 1, 
// and checking if that 1 is at the head of the array. 
// TODO: check if this goes against the standard?
#define IS_LE (((union {uint32_t num; uint8_t bytes[sizeof(uint32_t)]; }){num : 0x1}).bytes[0] == 1)
#define IS_REAL_FIRST ( offsetof(MyComplex, re) == 0 )

typedef struct MyComplex
{
    float re;
    float im;
} MyComplex;

static inline MyComplex MyComplex_zero()
{
    return (MyComplex){re : 0.0, im : 0.0};
}

static inline MyComplex MyComplex_im(float im)
{
    return (MyComplex){re : 0.0, im : im};
}

static inline float MyComplex_mag_2(MyComplex self) {
    return self.re * self.re + self.im * self.im;
}

static inline void MyComplex_read(uint8_t * raw_bytes, size_t bytes_len, MyComplex * outpt, size_t max_out) {
    size_t to_copy = max_out <= bytes_len/sizeof(MyComplex) ? max_out : bytes_len/sizeof(MyComplex); 
    if(IS_LE && IS_REAL_FIRST) {
        size_t bytes = sizeof(MyComplex) * to_copy;
        memcpy(outpt, raw_bytes, bytes);
        return; 
    }
    else {
        fprintf(stderr, "ERROR at %s %d: TODO\n", __FILE__, __LINE__);
        exit(1);
    }


}

static inline void MyComplex_write(MyComplex * src, size_t num, uint8_t *output, size_t outlen)
{
    size_t to_copy = outlen >= sizeof(MyComplex) * num ? num : outlen/sizeof(MyComplex); 
    fprintf(stderr, "%ld , %ld => %ld\n", outlen, num, to_copy);
    if(IS_LE && IS_REAL_FIRST) {
        size_t bytes = sizeof(MyComplex) * to_copy;
        memcpy(output, src, bytes);
        return; 
    }
    for(int idx = 0; idx < to_copy; idx+=1) {
        size_t offset = sizeof(MyComplex) * idx; 
        if(IS_LE) {
            memcpy(output + offset + 0, &src[idx].re, 4);
            memcpy(output + offset + 4, &src[idx].im, 4);
        }
        else {
            uint8_t * reptr = (uint8_t *) &src[idx].re;
            uint8_t * imptr = (uint8_t *) &src[idx].im;
            uint8_t * outptr = output + offset;
            for(int fltidx = 3; fltidx <= 0; fltidx --) {
                *outptr = reptr[fltidx];
                outptr += 1;
            }
            for(int fltidx = 3; fltidx <= 0; fltidx --) {
                *outptr = imptr[fltidx];
                outptr += 1;
            }
        }
    }
}