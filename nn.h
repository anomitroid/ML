#ifndef NN_HEADER 
#define NN_HEADER

// ----- libraries ------
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
// ----------------------

// ----- standard macros -----
#ifndef NN_MALLOC
#include <malloc.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT
// ---------------------------

// ----- custom macros -----
#define MATRIX_AT(m, i, j) ((m).elements[(m).stride * i + j])
#define MATRIX_DISPLAY(m) matrix_display(m, #m, 0)
#define NN_DISPLAY(nn) nn_display(nn, #nn)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define NN_INPUT(nn) ((nn).inputs[0])
#define NN_OUTPUT(nn) ((nn).inputs[(nn).count])
// -------------------------


// ----- utility functions -----
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}
// -----------------------------


// ----- matrix structure -----
typedef struct {
    size_t rows;                // number of rows
    size_t cols;                // number of columns
    size_t stride;              // number of elements to skip to get to next row (not always equal to cols)
    float* elements;            // continuous array of floats
} matrix;
// ----------------------------


// ----- matrix methods declaration -----
matrix matrix_alloc(size_t rows, size_t cols, size_t stride);
void matrix_display(matrix m, const char* name, size_t padding);
void matrix_randomise(matrix m, float low, float high);
void matrix_fill(matrix m, float x);
void matrix_multiplication(matrix destination, matrix m1, matrix m2);
void matrix_addition(matrix destination, matrix m);
void matrix_sigmoid(matrix m);
matrix matrix_row(matrix m, size_t i);
void matrix_copy(matrix destination, matrix source);
matrix matrix_data_alloc(float* data, size_t rows, size_t cols, size_t stride);
void matrix_free(matrix* m);
// -------------------------------------


// ----- NN structure ------
typedef struct {
    size_t count;
    matrix* weights;
    matrix* biases;
    matrix* inputs;
} NN;
// -------------------------


// ----- nn methods declaration -----
NN nn_alloc(size_t* architecture, size_t layer_count);
void nn_display(NN nn, const char* name);
void nn_randomise(NN nn, float low, float high);
void nn_free(NN* nn);
void nn_forward(NN nn);
float nn_cost(NN nn, matrix ti, matrix to);
void nn_finite_difference(NN nn, NN* g, float eps, matrix ti, matrix to);
void nn_learn(NN nn, NN g, float rate);
void nn_backprop(NN nn, NN* g, matrix ti, matrix to);
void nn_render_to_png(NN nn, int width, int height, const char* filename, float cost);
// ----------------------------------

#endif // NN_HEADER

#ifdef NN_IMPLEMENTATION

// ----- matrix methods definition -----
matrix matrix_alloc(size_t rows, size_t cols, size_t stride) {
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = stride;
    m.elements = NN_MALLOC(sizeof(*m.elements) * rows * cols);
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(rows > 0 && cols > 0 && stride > 0);
    return m;
}

void matrix_display(matrix m, const char* name, size_t padding) {
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(m.rows > 0 && m.cols > 0 && m.stride > 0);
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s", 2 * (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f  ", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

void matrix_randomise(matrix m, float low, float high) {
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(m.rows > 0 && m.cols > 0 && m.stride > 0);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            NN_ASSERT(i < m.rows && j < m.cols);
            MATRIX_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void matrix_fill(matrix m, float x) {
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(m.rows > 0 && m.cols > 0 && m.stride > 0);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            NN_ASSERT(i < m.rows && j < m.cols);
            MATRIX_AT(m, i, j) = x;
        }
    }
}

void matrix_multiplication(matrix destination, matrix m1, matrix m2) {
    NN_ASSERT(destination.elements != NULL && m1.elements != NULL && m2.elements != NULL);
    NN_ASSERT(destination.rows > 0 && destination.cols > 0 && destination.stride > 0);
    NN_ASSERT(m1.rows > 0 && m1.cols > 0 && m1.stride > 0);
    NN_ASSERT(m2.rows > 0 && m2.cols > 0 && m2.stride > 0);
    NN_ASSERT(m1.cols == m2.rows);
    NN_ASSERT(destination.rows == m1.rows);
    NN_ASSERT(destination.cols == m2.cols);
    for (size_t i = 0; i < destination.rows; i++) {
        for (size_t j = 0; j < destination.cols; j++) {
            MATRIX_AT(destination, i, j) = 0;
            for (size_t k = 0; k < m1.cols; k++) {
                MATRIX_AT(destination, i, j) += (MATRIX_AT(m1, i, k) * MATRIX_AT(m2, k, j));
            }
        }
    }
}

void matrix_addition(matrix destination, matrix m) {
    NN_ASSERT(destination.elements != NULL && m.elements != NULL);
    NN_ASSERT(destination.rows > 0 && destination.cols > 0 && destination.stride > 0);
    NN_ASSERT(m.rows > 0 && m.cols > 0 && m.stride > 0);
    NN_ASSERT(destination.rows == m.rows);
    NN_ASSERT(destination.cols == m.cols);
    for (size_t i = 0; i < destination.rows; i++) {
        for (size_t j = 0; j < destination.cols; j++) {
            NN_ASSERT(i < destination.rows && j < destination.cols);
            NN_ASSERT(i < m.rows && j < m.cols);
            MATRIX_AT(destination, i, j) += MATRIX_AT(m, i, j);
        }
    }
}

void matrix_sigmoid(matrix m) {
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(m.rows > 0 && m.cols > 0 && m.stride > 0);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}

matrix matrix_row(matrix m, size_t i) {
    NN_ASSERT(m.elements != NULL);
    NN_ASSERT(i < m.rows);
    return (matrix) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .elements = &MATRIX_AT(m, i, 0)
    };
}

void matrix_copy(matrix destination, matrix source) {
    NN_ASSERT(destination.elements != NULL && source.elements != NULL);
    NN_ASSERT(destination.rows == source.rows && destination.cols == source.cols);
    for (size_t i = 0; i < destination.rows; i++) {
        for (size_t j = 0; j < destination.cols; j++) {
            MATRIX_AT(destination, i, j) = MATRIX_AT(source, i, j);
        }
    }
}

matrix matrix_data_alloc(float* data, size_t rows, size_t cols, size_t stride) {
    NN_ASSERT(data != NULL);
    NN_ASSERT(rows > 0 && cols > 0 && stride > 0);
    return (matrix) {
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .elements = data
    };
}

void matrix_free(matrix* m) {
    NN_ASSERT(m != NULL);
    NN_ASSERT(m->elements != NULL);
    free(m -> elements);
    m -> elements = NULL;
}
// -------------------------------------


// ------- nn methods definition -------
NN nn_alloc(size_t* architecture, size_t layer_count) {
    NN nn; 
    nn.count = layer_count - 1;

    nn.inputs = NN_MALLOC(layer_count * sizeof(matrix));
    NN_ASSERT(nn.inputs != NULL);
    nn.weights = NN_MALLOC((layer_count - 1) * sizeof(matrix));
    NN_ASSERT(nn.weights != NULL);
    nn.biases = NN_MALLOC((layer_count - 1) * sizeof(matrix));
    NN_ASSERT(nn.biases != NULL);

    nn.inputs[0] = matrix_alloc(1, architecture[0], architecture[0]);
    for (size_t i = 1; i < layer_count; i++) {
        nn.inputs[i] = matrix_alloc(nn.inputs[i - 1].rows, architecture[i], architecture[i]);
        nn.weights[i - 1] = matrix_alloc(nn.inputs[i - 1].cols, architecture[i], architecture[i]);
        nn.biases[i - 1] = matrix_alloc(nn.inputs[i - 1].rows, architecture[i], architecture[i]);
    }
    return nn;
}

void nn_display(NN nn, const char* name) {
    printf("%s = [\n", name);
    char wname[32], bname[32];
    for (size_t i = 0; i < nn.count; i++) {
        snprintf(wname, sizeof(wname), "ws%zu", i);
        snprintf(bname, sizeof(bname), "bs%zu", i);
        matrix_display(nn.weights[i], wname, 4);
        matrix_display(nn.biases[i], bname, 4);
    }
    printf("]\n");
}

void nn_randomise(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; i++) {
        matrix_randomise(nn.weights[i], low, high);
        matrix_randomise(nn.biases[i], low, high);
    }
} 

void nn_free(NN* nn) {
    NN_ASSERT(nn != NULL);
    for (size_t i = 0; i < nn -> count; i++) {
        matrix_free(&nn -> weights[i]);
        matrix_free(&nn -> biases[i]);
        matrix_free(&nn -> inputs[i]);
    }
    matrix_free(&nn -> inputs[nn -> count]);
    free(nn -> inputs);
    free(nn -> weights);
    free(nn -> biases);
    nn -> inputs = NULL;
    nn -> weights = NULL;
    nn -> biases = NULL;
    nn -> count = 0;
}

void nn_forward(NN nn) {
    NN_ASSERT(nn.inputs != NULL && nn.weights != NULL && nn.biases != NULL);
    for (size_t i = 0; i < nn.count; i++) {
        matrix_multiplication(nn.inputs[i + 1], nn.inputs[i], nn.weights[i]);
        matrix_addition(nn.inputs[i + 1], nn.biases[i]);
        matrix_sigmoid(nn.inputs[i + 1]);
    }
}

float nn_cost(NN nn, matrix ti, matrix to) {
    NN_ASSERT(ti.elements != NULL && to.elements != NULL);
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    float result = 0;
    size_t n = ti.rows;
    for (size_t i = 0; i < n; i++) {
        matrix x = matrix_row(ti, i);
        matrix y = matrix_row(to, i);
        matrix_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q =  to.cols;
        for (size_t j = 0; j < q; j++) {
            float d = MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
            result += d * d;
        }
    }
    return result / n;
}

void nn_finite_difference(NN nn, NN* g, float eps, matrix ti, matrix to) {
    NN_ASSERT(g -> weights != NULL && g -> biases != NULL);
    NN_ASSERT(nn.weights != NULL && nn.biases != NULL);
    float c = nn_cost(nn, ti, to);
    float saved;
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.weights[i].rows; j++) {
            for (size_t k = 0; k < nn.weights[i].cols; k++) {
                saved = MATRIX_AT(nn.weights[i], j, k);
                MATRIX_AT(nn.weights[i], j, k) += eps;
                MATRIX_AT(g -> weights[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MATRIX_AT(nn.weights[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < nn.biases[i].rows; j++) {
            for (size_t k = 0; k < nn.biases[i].cols; k++) {
                saved = MATRIX_AT(nn.biases[i], j, k);
                MATRIX_AT(nn.biases[i], j, k) += eps;
                MATRIX_AT(g -> biases[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MATRIX_AT(nn.biases[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate) {
    NN_ASSERT(nn.weights != NULL && nn.biases != NULL && g.weights != NULL && g.biases != NULL);
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.weights[i].rows; j++) {
            for (size_t k = 0; k < nn.weights[i].cols; k++) {
                MATRIX_AT(nn.weights[i], j, k) -= rate * MATRIX_AT(g.weights[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.biases[i].rows; j++) {
            for (size_t k = 0; k < nn.biases[i].cols; k++) {
                MATRIX_AT(nn.biases[i], j, k) -= rate * MATRIX_AT(g.biases[i], j, k);
            }
        }
    }
}

void nn_backprop(NN nn, NN* g, matrix ti, matrix to) {
    NN_ASSERT(g -> weights != NULL && g -> biases != NULL);
    NN_ASSERT(nn.weights != NULL && nn.biases != NULL);
    NN_ASSERT(ti.elements != NULL && to.elements != NULL);
    NN_ASSERT(ti.rows == to.rows);

    size_t N = ti.rows;  
    size_t L = nn.count;  

    for (size_t l = 0; l < L; l++) {
        matrix_fill(g -> weights[l], 0.0f);
        matrix_fill(g -> biases [l], 0.0f);
    }

    matrix* deltas = NN_MALLOC((L + 1) * sizeof(matrix));
    NN_ASSERT(deltas != NULL);
    for (size_t l = 1; l <= L; l++) {
        size_t m_l = nn.inputs[l].cols;  
        deltas[l] = matrix_alloc(1, m_l, m_l);
    }

    for (size_t ex = 0; ex < N; ex++) {
        matrix xrow = matrix_row(ti, ex);  
        matrix_copy(nn.inputs[0], xrow);
        nn_forward(nn);

        matrix a_L = nn.inputs[L];
        matrix yrow = matrix_row(to, ex);
        matrix delta_L = deltas[L];

        size_t m_L = a_L.cols;
        for (size_t k = 0; k < m_L; k++) {
            float a_val = MATRIX_AT(a_L, 0, k);
            float y_val = MATRIX_AT(yrow, 0, k);
            float dL_da  = 2.0f * (a_val - y_val);
            float da_dz  = a_val * (1.0f - a_val);
            MATRIX_AT(delta_L, 0, k) = dL_da * da_dz;
        }

        for (size_t layer = L; layer > 1; layer--) {
            size_t m_prev = nn.inputs[layer - 1].cols;
            size_t m_cur = nn.inputs[layer].cols;

            matrix W_next = nn.weights[layer - 1];
            matrix delta_next = deltas[layer];
            matrix a_prev = nn.inputs[layer - 1];
            matrix delta_cur = deltas[layer - 1];

            for (size_t j = 0; j < m_prev; j++) {
                float a_val = MATRIX_AT(a_prev, 0, j);
                float da_dz = a_val * (1.0f - a_val);
                float sum = 0.0f;
                for (size_t k = 0; k < m_cur; k++) {
                    float dnext_val = MATRIX_AT(delta_next, 0, k);
                    float w_val = MATRIX_AT(W_next, j, k);
                    sum += dnext_val * w_val;
                }
                MATRIX_AT(delta_cur, 0, j) = sum * da_dz;
            }
        }

        for (size_t layer = 1; layer <= L; layer++) {
            matrix delta_l = deltas[layer];
            matrix a_prev  = nn.inputs[layer - 1];

            size_t m_prev = a_prev.cols;
            size_t m_cur  = delta_l.cols;

            for (size_t j = 0; j < m_prev; j++) {
                float a_val = MATRIX_AT(a_prev, 0, j);
                for (size_t k = 0; k < m_cur; k++) {
                    float d_val = MATRIX_AT(delta_l, 0, k);
                    MATRIX_AT(g -> weights[layer - 1], j, k) += a_val * d_val;
                }
            }
            for (size_t k = 0; k < m_cur; k++) {
                float d_val = MATRIX_AT(delta_l, 0, k);
                MATRIX_AT(g -> biases[layer - 1], 0, k) += d_val;
            }
        }
    }

    for (size_t layer = 0; layer < L; layer++) {
        matrix Gw = g -> weights[layer];
        for (size_t i = 0; i < Gw.rows; i++) {
            for (size_t j = 0; j < Gw.cols; j++) {
                MATRIX_AT(Gw, i, j) /= (float)N;
            }
        }
        matrix Gb = g -> biases[layer];
        for (size_t j = 0; j < Gb.cols; j++) {
            MATRIX_AT(Gb, 0, j) /= (float)N;
        }
    }

    for (size_t l = 1; l <= L; l++) {
        matrix_free(&deltas[l]);
    }
    free(deltas);
}

// -------------------------------------

#endif // NN_IMPLEMENTATION


#ifdef NN_RENDER_IMPLEMENTATION

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static uint32_t hex24_to_argb(uint32_t hex24) {
    uint32_t r = (hex24 >> 16) & 0xFF;
    uint32_t g = (hex24 >> 8) & 0xFF;
    uint32_t b = hex24 & 0xFF;
    return (0xFF << 24) | (b << 16) | (g << 8) | (r);
}

static uint8_t lerp8(uint8_t a, uint8_t b, float t) {
    return (uint8_t)roundf(a + (b - a) * t);
}

static uint32_t lerp_color(uint32_t c0, uint32_t c1, float t) {
    uint8_t r0 =  c0 & 0xFF;
    uint8_t g0 = (c0 >> 8) & 0xFF;
    uint8_t b0 = (c0 >> 16) & 0xFF;
    uint8_t a0 = (c0 >> 24) & 0xFF;
    uint8_t r1 =  c1 & 0xFF;
    uint8_t g1 = (c1 >> 8) & 0xFF;
    uint8_t b1 = (c1 >> 16) & 0xFF;
    uint8_t a1 = (c1 >> 24) & 0xFF;
    uint8_t r = lerp8(r0, r1, t);
    uint8_t g = lerp8(g0, g1, t);
    uint8_t b = lerp8(b0, b1, t);
    uint8_t a = lerp8(a0, a1, t);
    return (a << 24) | (b << 16) | (g << 8) | (r);
}

void nn_render_to_png(NN nn, int width, int height, const char* filename, float cost) {
    uint32_t *pixels = (uint32_t*)malloc(sizeof(uint32_t) * width * height);
    Olivec_Canvas img = olivec_canvas(pixels, width, height, width);

    const size_t NG = 9;
    uint32_t neuron_grad[NG] = {
        hex24_to_argb(0x005F73), // #005F73
        hex24_to_argb(0x0A9396), // #0A9396
        hex24_to_argb(0x94D2BD), // #94D2BD
        hex24_to_argb(0xE9D8A6), // #E9D8A6
        hex24_to_argb(0xEE9B00), // #EE9B00
        hex24_to_argb(0xCA6702), // #CA6702
        hex24_to_argb(0xBB3E03), // #BB3E03
        hex24_to_argb(0xAE2012), // #AE2012
        hex24_to_argb(0x9B2226), // #9B2226
    };

    const size_t CG = 5;
    uint32_t conn_grad[CG] = {
        hex24_to_argb(0xE0B1CB), // #E0B1CB
        hex24_to_argb(0xBE95C4), // #BE95C4
        hex24_to_argb(0x9F86C0), // #9F86C0
        hex24_to_argb(0x5E548E), // #5E548E
        hex24_to_argb(0x231942), // #231942
    };

    uint32_t background_color = hex24_to_argb(0xCCCCFF); // #CCCCFF 
    uint32_t input_color = hex24_to_argb(0x7B68EE); // #7B68EE  
    uint32_t frame_color = hex24_to_argb(0x2B4593); // #2B4593  
    uint32_t text_color = hex24_to_argb(0x000000); // #000000

    olivec_fill(img, background_color);

    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = width - 2 * layer_border_hpad;
    int nn_height = height - 2 * layer_border_vpad;
    int nn_x = width / 2 - nn_width / 2;
    int nn_y = height / 2 - nn_height / 2;

    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;

    for (size_t l = 0; l < arch_count; l++) {
        int neurons_in = nn.inputs[l].cols;
        int layer_vpad = nn_height / neurons_in;

        for (size_t i = 0; i < (size_t)neurons_in; i++) {
            int cx1 = nn_x + (int)l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + (int)i * layer_vpad + layer_vpad / 2;

            if (l + 1 < arch_count) {
                int neurons_out = nn.inputs[l + 1].cols;
                int next_vpad  = nn_height / neurons_out;

                for (size_t j = 0; j < (size_t)neurons_out; j++) {
                    int cx2 = nn_x + (int)(l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + (int)j * next_vpad + next_vpad / 2;

                    float w = MATRIX_AT(nn.weights[l], i, j);
                    float sw = 1.f / (1.f + expf(-w));
                    if (sw < 0.f) sw = 0.f;
                    else if (sw > 1.f) sw = 1.f;

                    float idxf = sw * (CG - 1);
                    size_t idx0 = (size_t)floorf(idxf);
                    size_t idx1 = (idx0 < CG - 1 ? idx0 + 1 : idx0);
                    float frac = idxf - (float)idx0;

                    uint32_t col_line = lerp_color(conn_grad[idx0], conn_grad[idx1], frac);
                    olivec_line(img, cx1, cy1, cx2, cy2, col_line);
                }
            }

            uint32_t neuron_color;
            if (l == 0) {
                neuron_color = input_color;
            } else {
                float raw = MATRIX_AT(nn.biases[l - 1], 0, i);
                float s = 1.f / (1.f + expf(-raw));
                if (s < 0.f) s = 0.f;
                else if (s > 1.f) s = 1.f;

                float idxf = s * (NG - 1);
                size_t idx0 = (size_t)floorf(idxf);
                size_t idx1 = (idx0 < NG - 1 ? idx0 + 1 : idx0);
                float frac = idxf - (float)idx0;

                neuron_color = lerp_color(neuron_grad[idx0], neuron_grad[idx1], frac);
            }

            olivec_circle(img, cx1, cy1, neuron_radius + 1.5, input_color);
            olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
        }
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "Cost: %.4f", cost);
    // olivec_text(img, buf, 10, 10, olivec_default_font, 16, text_color);

    uint32_t frame_thicc = 10;
    olivec_frame(img, 0, 0, width - 1, height - 1, frame_thicc, frame_color);

    stbi_write_png(filename, width, height, 4, img.pixels, img.stride * sizeof(uint32_t));

    free(pixels);
}

#endif // NN_RENDER_IMPLEMENTATION


/*
Neural Network

            w11                    w31
    x1  ●─────────────●  h1  ●─────────────●  y
        │\           /│ +b1  │            /│ +b3
        │ \         / │      │           / │
        │  \  w12  /  │      │   w32    /  │
        │   \     /   │      │         /   │
        │    \   /    │      │        /    │
        │     \ /     │      │       /     │
        │      ×      │      │      /      │
        │     / \     │      │     /       │
        │    /   \    │      │    /        │
        │   /     \   │      │   /         │
        │  /  w21  \  │      │  /          │
        │ /         \ │      │ /           │
        │/           \│ +b2  │/            │
    x2  ●─────────────●  h2  ●─────────────●
                w22


x1, x2 are the input neurons. (1st layer)
h1, h2 are the hidden layer. (2nd layer)
y is the output neuron. (3rd layer)

how are we going to get the values of h1 and h2?

we know, 
    h1 = sig (x1 * w11 + x2 * w21 + b1)
    h2 = sig (x2 * w22 + x1 * w12 + b2)

similarly, in case of finding y
    y = sig (h1 * w31 + h2 * w32 + b3)


when we look closely, it is clear that the expressions of the form 
    a1 * w1 + a2 * w2 
look really similar to the results of dot products
i.e. matrix mulriplications

so, we try to model the values of h1, h2, and all subsequent neurons as dot products
and for that, we need to represent the weights as matrices

so, basically, for each layer, we have
    1. input matrix
    2. weights matrix
    3. bias matrix
all we have to do is to find the dimensions of these matrices

for layer 1-2
    input is 2 elements (x1, x2)
    output is also 2 elements (h1, h2)
    4 weights are involved (w11, w12, w21, w22)
    biases are 2 (b1, b2)

    so, if input matrix is of dimension 1 * 2
        i.e. [ x1   x1 ]
        if the weights matrix is of dimension 2 * 2
        i.e. [ w11  w12 ]
             [ w21  w22 ]
        then, output will be of dimension 1 * 2
        i.e. [ h1   h2 ]
        then, we can add bias matrix to this
        so, bias will be of dimension 1 * 2
        i.e. [ b1   b2 ]
        after addind them, we apply sigmoid on each element of the output matrix

    Let's try this out and see if the results match

    [ x1    x2 ] * [ w11  w12 ] + [ b1  b2 ]
                   [ w21  w22 ]

    = [ x1 * w11 + x2 * w21     x1 * w12 + x2 * w22 ] + [ b1    b2 ]
    = [ x1 * w11 + w2 * w21 + b1    x1 * w12 + x2 * w22 + b2 ]
    then, sigmoid on each element of this matrix
    = [ sig(x1 * w11 + w2 * w21 + b1)    sig(x1 * w12 + x2 * w22 + b2) ]

    as we can see, the [0, 0] element is h1
                       [0, 1] element is h2

    let us try doing the same for layers 2-3

    now, input = [ h1   h2 ]
         weights = [ w31 ]
                   [ w32 ]
         output = [ y ]
         bias = [ b3 ]

    we do
        input * weights + bias
        
        = [ h1 * w31 + h2 * w32 ] + [ b3 ]
        = [ h1 * w31 + h2 * w32 + b3 ]

    then signoid of the element
        = [ sig(h1 * w31 + h2 * w32 + b3) ]

thus, we have generalised on the 'forward' formula i.e. the transition between 2 layers

next layer = sigmoid(inputs * weights + biases)


*/