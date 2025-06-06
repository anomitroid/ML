#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdint.h>
#include <math.h>
#include <time.h>

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_HEIGHT * IMG_WIDTH];

static uint32_t hex24_to_argb(uint32_t hex24) {
    uint32_t r = (hex24 >> 16) & 0xFF;
    uint32_t g = (hex24 >>  8) & 0xFF;
    uint32_t b =  hex24 & 0xFF;
    return (0xFF << 24) | (b << 16) | (g << 8) | (r);
}

static uint8_t lerp8(uint8_t a, uint8_t b, float t) {
    return (uint8_t)roundf(a + (b - a) * t);
}

static uint32_t lerp_color(uint32_t c0, uint32_t c1, float t) {
    uint8_t r0 =  c0 & 0xFF;
    uint8_t g0 = (c0 >>  8) & 0xFF;
    uint8_t b0 = (c0 >> 16) & 0xFF;
    uint8_t a0 = (c0 >> 24) & 0xFF;
    uint8_t r1 =  c1 & 0xFF;
    uint8_t g1 = (c1 >>  8) & 0xFF;
    uint8_t b1 = (c1 >> 16) & 0xFF;
    uint8_t a1 = (c1 >> 24) & 0xFF;
    uint8_t r = lerp8(r0, r1, t);
    uint8_t g = lerp8(g0, g1, t);
    uint8_t b = lerp8(b0, b1, t);
    uint8_t a = lerp8(a0, a1, t);
    return (a << 24) | (b << 16) | (g << 8) | (r);
}

void nn_render(Olivec_Canvas img, NN nn) {
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
        hex24_to_argb(0x4cc9f0), // #4cc9f0
        hex24_to_argb(0x4361ee), // #4361ee
        hex24_to_argb(0x3a0ca3), // #3a0ca3
        hex24_to_argb(0x7209b7), // #7209b7
        hex24_to_argb(0xf72585), // #f72585
    };

    uint32_t background_color = hex24_to_argb(0xCCCCFF); // #CCCCFF
    uint32_t input_color = hex24_to_argb(0x7B68EE); // #7B68EE  
    uint32_t connection_color = hex24_to_argb(0xB0B0B0); // #B0B0B0

    olivec_fill(img, background_color);

    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = img.width  - 2 * layer_border_hpad;
    int nn_height = img.height - 2 * layer_border_vpad;
    int nn_x = img.width  / 2 - nn_width  / 2;
    int nn_y = img.height / 2 - nn_height / 2;

    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;

    for (size_t l = 0; l < arch_count; l++) {
        // Vertical spacing for this layer
        int layer_vpad1 = nn_height / nn.inputs[l].cols;

        for (size_t i = 0; i < nn.inputs[l].cols; i++) {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;

            // Draw connections to next layer
            if (l + 1 < arch_count) {
                int layer_vpad2 = nn_height / nn.inputs[l + 1].cols;
                for (size_t j = 0; j < nn.inputs[l + 1].cols; j++) {
                    int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;

                    float w = MATRIX_AT(nn.weights[l], i, j);
                    float sw = sigmoidf(-w);
                    if (sw < 0.f) sw = 0.f;
                    else if (sw > 1.f) sw = 1.f;
                    float idxf = sw * (CG - 1);
                    size_t idx0 = (size_t)floorf(idxf);
                    size_t idx1 = idx0 < (CG - 1) ? (idx0 + 1) : idx0;
                    float frac = idxf - (float)idx0;
                    uint32_t connection_color = lerp_color(conn_grad[idx0], conn_grad[idx1], frac);

                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }

            // Choose neuron color
            uint32_t neuron_color;
            if (l == 0) {
                neuron_color = input_color;
            } else {
                float raw = MATRIX_AT(nn.biases[l - 1], 0, i);
                float s = sigmoidf(-raw); 
                if (s < 0.f) s = 0.f;
                else if (s > 1.f) s = 1.f;
                float idxf = s * (NG - 1);
                size_t idx0 = (size_t)floorf(idxf);
                size_t idx1 = idx0 < (NG - 1) ? (idx0 + 1) : idx0;
                float frac = idxf - (float)idx0;
                neuron_color = lerp_color(neuron_grad[idx0], neuron_grad[idx1], frac);
            }
            olivec_circle(img, cx1, cy1, neuron_radius + 1.5, input_color);
            olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
        }
    }
}


int main(void) {
    srand(time(0));

    size_t arch[] = {2, 5, 3, 5, 1};
    size_t arch_count = ARRAY_SIZE(arch);
    NN nn = nn_alloc(arch, arch_count);
    nn_randomise(nn, -1, 1);

    NN_DISPLAY(nn);

    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    nn_render(img, nn);

    uint32_t frame_thicc = 10;
    uint32_t frame_color = hex24_to_argb(0x2B4593); // #2B4593
    olivec_frame(img, 0, 0, img.width - 1, img.height - 1, frame_thicc, frame_color);

    const char* img_file_path = "nn.png";
    if (!stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride * sizeof(uint32_t))) {
        printf("ERROR: could not save file %s\n", img_file_path);
    }

    return 0;
}