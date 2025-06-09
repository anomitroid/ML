#define NN_IMPLEMENTATION
#include "nn.h"

int main(void) {
    matrix t = matrix_alloc(4, 3, 3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            size_t rows = i * 2 + j;
            MATRIX_AT(t, rows, 0) = i;
            MATRIX_AT(t, rows, 1) = j;
            MATRIX_AT(t, rows, 2) = i ^ j;
        }
    }

    const char* out_file_path = "xor.mat";
    FILE* out = fopen(out_file_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }

    matrix_save(out, t);
    fclose(out);
    
    printf("Generated %s\n", out_file_path);
    
    return 0;
}