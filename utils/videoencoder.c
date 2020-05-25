#include <libavcodec/avcodec.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdint.h>

#ifndef WIDTH
    #define WIDTH (100)
#endif
#ifndef HEIGHT
    #define HEIGHT (100)
#endif
#ifndef FRAMES
    #define FRAMES (500)
#endif

#define FRAME_SIZE (WIDTH * HEIGHT)
#define PIXEL_SIZE (3)
#define FRAME_BYTES (PIXEL_SIZE * FRAME_SIZE)

#define ASSERT(cond)                                                         \
    do                                                                       \
    {                                                                        \
        if (!(cond))                                                         \
        {                                                                    \
            fprintf(stderr, "ASSERTION " #cond " FAILED at %d\n", __LINE__); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define KILL_ON_ERR(expr)                                                                   \
    do                                                                                      \
    {                                                                                       \
        int code = (expr);                                                                  \
        if (code < 0)                                                                       \
        {                                                                                   \
            fprintf(stderr, "Line %d: " #expr " returned error code %d\n", __LINE__, code); \
            exit(1);                                                                        \
        }                                                                                   \
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
void write_frame_ppm(int idx, uint8_t * frame_rgb) {
    char idxbuff[32];
    snprintf(idxbuff, 32, "frame%03d.ppm", idx);
    FILE *output;
    KILL_ON_NULL(output = fopen(idxbuff, "wb"));
    fprintf(output, "P6\n");
    fprintf(output, "%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(frame_rgb, 1, FRAME_BYTES, output);
    fclose(output);
}
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "USAGE: videoencoder <input file> <output file>\n");
        exit(1);
    }
    char *input_file = argv[1];
    int input_fd;
    KILL_ON_ERR(input_fd = open(input_file, O_RDONLY));
    struct stat sb;
    KILL_ON_ERR(fstat(input_fd, &sb));
    uint8_t *addr;
    KILL_ON_NULL(addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, input_fd, 0));
    char *output_file = argv[2];
    AVCodec *codec;
    KILL_ON_NULL(codec = avcodec_find_encoder_by_name("libx264rgb"));
    printf("Using codec: %s \n", codec->long_name);
    AVCodecContext *ctx;
    KILL_ON_NULL(ctx = avcodec_alloc_context3(codec));
    AVPacket *pkt;
    KILL_ON_NULL(pkt = av_packet_alloc());
    ctx->width = WIDTH;
    ctx->height = HEIGHT;

    ctx->framerate = (AVRational){1, 1};
    ctx->time_base = (AVRational){ctx->framerate.den, ctx->framerate.num};
    ASSERT(ctx->framerate.den == ctx->time_base.num && ctx->framerate.num == ctx->time_base.den);
    ctx->pix_fmt = AV_PIX_FMT_RGB24;

    KILL_ON_ERR(avcodec_open2(ctx, codec, NULL));

    FILE *output;
    KILL_ON_NULL(output = fopen(argv[2], "wb"));
    AVFrame *frame;
    KILL_ON_NULL(frame = av_frame_alloc());
    frame->format = ctx->pix_fmt;
    frame->width = ctx->width;
    frame->height = ctx->height;
    KILL_ON_ERR(av_frame_get_buffer(frame, 1));
    fprintf(stderr, "A\n");
    fprintf(stderr, "%p %p %s %d\n", frame->data[0], frame->buf[0]->data,  frame->data[0] == frame->buf[0]->data ? "T" : "F",  frame->buf[0]->size);
    for(int kkk = 0; kkk < AV_NUM_DATA_POINTERS; kkk++) {
        fprintf(stderr, "%d ", frame->linesize[kkk]);
    }
    fprintf(stderr, "\n");
    for (int cur_tidx = 0; cur_tidx < FRAMES; cur_tidx++)
    {
        uint8_t *frame_start = addr + (cur_tidx * FRAME_BYTES);
        #ifdef DBG_WRITE_FRAMES
        write_frame_ppm(cur_tidx, frame_start);
        #endif
        KILL_ON_ERR(av_frame_make_writable(frame));
        memcpy(frame->data[0], frame_start, FRAME_BYTES);
        frame->pts = cur_tidx;

        KILL_ON_ERR(avcodec_send_frame(ctx, frame));
        while (1)
        {
            int recv_code = avcodec_receive_packet(ctx, pkt);
            if (recv_code == AVERROR(EAGAIN) || recv_code == AVERROR_EOF)
            {
                break;
            }
            KILL_ON_ERR(recv_code);
            fwrite(pkt->data, 1, pkt->size, output);
            av_packet_unref(pkt);
        }
    }

    KILL_ON_ERR(avcodec_send_frame(ctx, NULL));
    while (1)
    {
        int recv_code = avcodec_receive_packet(ctx, pkt);
        if (recv_code == AVERROR(EAGAIN) || recv_code == AVERROR_EOF)
        {
            break;
        }
        KILL_ON_ERR(recv_code);
        fwrite(pkt->data, 1, pkt->size, output);
        av_packet_unref(pkt);
    }
    fclose(output);
    avcodec_free_context(&ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    KILL_ON_ERR(munmap(addr, sb.st_size));
    return 0;
}
