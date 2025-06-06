#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void) {
    srand(time(0));

    size_t n = (1 << BITS);
    size_t rows = n * n;
    matrix ti = matrix_alloc(rows, 2 * BITS, 2 * BITS);
    matrix to = matrix_alloc(rows, BITS + 1, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; j++) {
            MATRIX_AT(ti, i, j) = (x >> j) & 1;
            MATRIX_AT(ti, i, j + BITS) = (y >> j) & 1;
            MATRIX_AT(to, i, j) = (z >> j) & 1;
        }
        MATRIX_AT(to, i, BITS) = z >= n;
    }

    size_t architecture[] = {2 * BITS, 4 * BITS, BITS + 1};

    NN nn = nn_alloc(architecture, ARRAY_SIZE(architecture));
    NN g = nn_alloc(architecture, ARRAY_SIZE(architecture));

    nn_randomise(nn, 0, 1);

    float rate = 1;

    printf("Initial Cost = %f\n", nn_cost(nn, ti, to));
    
    for (size_t iter = 0; iter < 10 * 1000; iter++) {
        #if 1
            nn_backprop(nn, &g, ti, to);
        #else
            nn_finite_difference(nn, &g, 1e-1, ti, to);
        #endif
        nn_learn(nn, g, rate);
        // if (iter % 10000 == 0) {
            printf("%zu: Cost = %f\n", iter, nn_cost(nn, ti, to));
        // }
    }

    printf("Final Cost = %f\n", nn_cost(nn, ti, to));
    printf("---------------------------------------\n");

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