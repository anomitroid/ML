#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void) {
    size_t n = (1 << BITS);
    size_t rows = n * n;
    matrix t = matrix_alloc(rows, 2 * BITS + BITS + 1, 2 * BITS + BITS + 1);
    matrix ti = {
        .elements = &MATRIX_AT(t, 0, 0),
        .rows = t.rows,
        .cols = 2 * BITS,
        .stride = t.stride,
    };
    matrix to = {
        .elements = &MATRIX_AT(t, 0, 2 * BITS),
        .rows = t.rows,
        .cols = BITS + 1,
        .stride = t.stride,
    };
    // matrix ti = matrix_alloc(rows, 2 * BITS, 2 * BITS);
    // matrix to = matrix_alloc(rows, BITS + 1, BITS + 1);

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

    const char* out_file_path = "adder.mat";
    FILE* out = fopen("adder.mat", "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }
    matrix_save(out, t);
    fclose(out);
    printf("Generated %s\n", out_file_path);
}