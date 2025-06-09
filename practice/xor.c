#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main(void) {
    srand(time(0));

    matrix ti = matrix_data_alloc(&td[0], 4, 2, 3);

    matrix to = matrix_data_alloc(&td[2], 4, 1, 3);

    size_t architecture[] = {2, 4, 1};

    NN nn = nn_alloc(architecture, ARRAY_SIZE(architecture));
    NN g = nn_alloc(architecture, ARRAY_SIZE(architecture));

    nn_randomise(nn, 0, 1);

    // float eps = 1e-4;
    float rate = 1e-2;

    printf("Initial Cost = %f\n", nn_cost(nn, ti, to));
    
    #if 1
    for (size_t iter = 0; iter < 5 * 1000 * 1000; iter++) {
        nn_backprop(nn, &g, ti, to);
        nn_learn(nn, g, rate);
        if (iter % 10000 == 0) {
            printf("Iter %zu: Cost = %f\n", iter, nn_cost(nn, ti, to));
        }
    }
    #endif

    #if 0
    for (size_t i = 0; i < 100 * 1000; i++) {
        nn_finite_difference(nn, &g, eps, ti, to);
        nn_learn(nn, g, rate);
    }
    #endif


    printf("Final Cost = %f\n", nn_cost(nn, ti, to));
    printf("---------------------------------------\n");

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y = MATRIX_AT(NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }    

    nn_free(&nn);
    nn_free(&g);

    return 0;
}