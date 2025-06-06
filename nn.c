#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define NN_IMPLEMENTATION
#define NN_RENDER_IMPLEMENTATION
#include "nn.h"

#define BITS 4
#define IMG_WIDTH 800
#define IMG_HEIGHT 600

int main(void) {
    srand((unsigned)time(NULL));

    size_t n = (1 << BITS);
    size_t rows = n * n;
    matrix ti = matrix_alloc(rows, 2 * BITS, 2 * BITS);
    matrix to = matrix_alloc(rows, BITS + 1, BITS + 1);

    for (size_t i = 0; i < rows; i++) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; j++) {
            MATRIX_AT(ti, i, j) = (float)((x >> j) & 1);
            MATRIX_AT(ti, i, j + BITS) = (float)((y >> j) & 1);
        }
        for (size_t j = 0; j < BITS; j++) {
            MATRIX_AT(to, i, j) = (float)((z >> j) & 1);
        }
        MATRIX_AT(to, i, BITS) = (float)(z >= n);
    }

    size_t architecture[] = { 2 * BITS, 4 * BITS, BITS + 1 };
    NN nn = nn_alloc(architecture, ARRAY_SIZE(architecture));
    NN g = nn_alloc(architecture, ARRAY_SIZE(architecture));
    nn_randomise(nn, -1, 1);

    float rate = 1.0f;
    printf("Initial Cost = %f\n", nn_cost(nn, ti, to));

    struct stat st = {0};
    if (stat("frames", &st) == -1) {
        mkdir("frames", 0755);
    }

    const size_t total_iters = 10 * 1000;
    const size_t frame_interval = 100;
    size_t frame_count = 0;

    for (size_t iter = 0; iter < total_iters; iter++) {
        nn_backprop(nn, &g, ti, to);
        nn_learn(nn, g, rate);

        if ((iter % frame_interval) == 0) {
            float c = nn_cost(nn, ti, to);

            char fname[64];
            snprintf(fname, sizeof(fname), "frames/frame_%05zu.png", frame_count++);

            nn_render_to_png(nn, IMG_WIDTH, IMG_HEIGHT, fname, c);

            printf("Wrote %s (iter=%zu, cost=%f)\n", fname, iter, nn_cost(nn, ti, to));
        }
    }

    printf("Final Cost = %f\n", nn_cost(nn, ti, to));
    printf("Generated %zu frames.\n", frame_count);

    nn_free(&nn);
    nn_free(&g);
    return 0;
}
