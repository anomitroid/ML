#include <time.h>
#include "raylib.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4
#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)

void nn_render_raylib(NN nn) {
    Color background_color = {0x18, 0x18, 0x18, 0xFF};
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00, 0xFF};

    ClearBackground(background_color);

    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = IMG_WIDTH - 2 * layer_border_hpad;
    int nn_height = IMG_HEIGHT - 2 * layer_border_vpad;
    int nn_x = IMG_WIDTH / 2 - nn_width / 2;
    int nn_y = IMG_HEIGHT / 2 - nn_height / 2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; l++) {
        int layer_vpad1 = nn_height / nn.inputs[l].cols;
        for (size_t i = 0; i < nn.inputs[l].cols; i++) {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch_count) {
                int layer_vpad2 = nn_height / nn.inputs[l + 1].cols;
                for (size_t j = 0; j < nn.inputs[l + 1].cols; j++) {
                    int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;
                    high_color.a = floorf(255.f * sigmoidf(MATRIX_AT(nn.weights[l], j, i)));
                    ColorAlphaBlend(low_color, high_color, WHITE);
                    DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f * sigmoidf(MATRIX_AT(nn.weights[l - 1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            }
            else {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

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

    NN_DISPLAY(nn);

    float rate = 1.0f;
    printf("Initial Cost = %f\n", nn_cost(nn, ti, to));

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "adder");
    SetTargetFPS(60);

    // struct stat st = {0};
    // if (stat("frames", &st) == -1) {
    //     mkdir("frames", 0755);
    // }

    const size_t total_iters = 10 * 1000;
    // const size_t frame_interval = 250;
    // size_t frame_count = 0;

    size_t iter = 0;
    while(!WindowShouldClose()) {
        // for (size_t iter = 0; iter < total_iters; iter++) {
        if (iter < total_iters) {
            nn_backprop(nn, &g, ti, to);
            nn_learn(nn, g, rate);
            printf("%zu: c = %f\n", iter, nn_cost(nn, ti, to));
            iter ++;

            // if ((iter % frame_interval) == 0) { 
            //     float c = nn_cost(nn, ti, to);

            //     char fname[64];
            //     snprintf(fname, sizeof(fname), "frames/frame_%05zu.png", frame_count++);

            //     nn_render_to_png(nn, IMG_WIDTH, IMG_HEIGHT, fname, c);

            //     printf("Wrote %s (iter=%zu, cost=%f)\n", fname, iter, nn_cost(nn, ti, to));
        }
        BeginDrawing();
        nn_render_raylib(nn);
        EndDrawing();
    }

    printf("Final Cost = %f\n", nn_cost(nn, ti, to));
    // printf("Generated %zu frames.\n", frame_count);

    size_t fails = 0;
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < n; y++) {
            size_t z = x + y;
            for (size_t j = 0; j < BITS; j++) {
                MATRIX_AT(NN_INPUT(nn), 0, j) = (x>>j)&1;
                MATRIX_AT(NN_INPUT(nn), 0, j + BITS) = (y>>j)&1;
            }
            nn_forward(nn);
            if (MATRIX_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
                if (z < n) {
                    printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
                    fails += 1;
                }
            } 
            else {
                size_t a = 0;
                for (size_t j = 0; j < BITS; j++) {
                    size_t bit = MATRIX_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                    a |= bit<<j;
                }
                if (z != a) {
                    printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
                    fails ++;
                }
            }
        }
    }

    if (fails == 0) printf("OK\n");

    nn_free(&nn);
    nn_free(&g);
    return 0;
}
