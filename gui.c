#include "raylib.h"
#include <time.h>
#include <float.h>

#define SV_IMPLEMENTATION
#include "sv.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define WINDOW_FACTOR 80
#define WINDOW_WIDTH (16 * WINDOW_FACTOR)
#define WINDOW_HEIGHT (9 * WINDOW_FACTOR)

typedef struct {
    size_t* items;
    size_t count;
    size_t capacity;
} Arch;

typedef struct {
    float* items;
    size_t count;
    size_t capacity;
} Cost_Plot;

#define DA_INIT_CAP 256
#define da_append(da, item) \
    do { \
        if ((da)->count >= (da)->capacity) { \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
            (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
            assert((da)->items != NULL && "More RAM Needed"); \
        } \
        (da)->items[(da)->count++] = (item); \
    } while (0)

char* args_shift(int* argc, char*** argv) {
    assert(*argc > 0);
    char* result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

void nn_render_raylib(NN nn, float rx, float ry, float rw, float rh) {
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00, 0xFF};

    int neuron_radius = rh * 0.03;
    float layer_border_vpad = rh * 0.08;
    float layer_border_hpad = rw * 0.06;
    float nn_width = rw - 2 * layer_border_hpad;
    float nn_height = rh - 2 * layer_border_vpad;
    float nn_x = rx + rw / 2 - nn_width / 2;
    float nn_y = ry + rh / 2 - nn_height / 2;
    size_t arch_count = nn.count + 1;
    float layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; l++) {
        float layer_vpad1 = nn_height / nn.inputs[l].cols;
        for (size_t i = 0; i < nn.inputs[l].cols; i++) {
            float cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            float cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch_count) {
                float layer_vpad2 = nn_height / nn.inputs[l + 1].cols;
                for (size_t j = 0; j < nn.inputs[l + 1].cols; j++) {
                    float cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    float cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;
                    float value = sigmoidf(MATRIX_AT(nn.weights[l], i, j));
                    high_color.a = floorf(255.f * value);
                    ColorAlphaBlend(low_color, high_color, WHITE);
                    float thick = rh * 0.002;
                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
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

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh) {
    float min = FLT_MAX, max = FLT_MIN;
    for (size_t i = 0; i < plot.count; i++) {
        if (max < plot.items[i]) max = plot.items[i];
        if (min > plot.items[i]) min = plot.items[i];
    }
    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 1000) n = 1000;
    for (size_t i = 0; i + 1 < plot.count; i++) {
        float x1 = rx + (float) rw / n * i;
        float y1 = ry + (1 - (plot.items[i] - min) / (max - min)) * rh;
        float x2 = rx + (float) rw / n * (i + 1);
        float y2 = ry + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;
        DrawLineEx((Vector2) {x1, y1}, (Vector2) {x2, y2}, rh * 0.005, RED);
    }
    float y0 = ry + (1 - (0 - min)/(max - min)) * rh;
    DrawLineEx((Vector2) {rx + 0, y0}, (Vector2) {rx + rw - 1, y0}, rh * 0.005, WHITE);
    DrawText("0", rx + 0, y0 - rh * 0.04, rh * 0.04, WHITE);
}


int main(int argc, char** argv) {
    srand(time(0));

    const char* program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: no architecture file was provided\n");
        return 1;
    }
    
    const char* arch_file_path = args_shift(&argc, &argv);
    
    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: no data file was provided\n");
        return 1;
    }
    
    const char* data_file_path = args_shift(&argc, &argv); 

    unsigned int buffer_len = 0;
    unsigned char* buffer = LoadFileData(arch_file_path, &buffer_len);
    if (buffer == NULL) {
        return 1;
    }

    String_View content = sv_from_parts((const char*) buffer, buffer_len);

    Arch arch = {0};

    content = sv_trim_left(content);
    while (content.count > 0 && content.data[0]) {
        int x = sv_chop_u64(&content);
        da_append(&arch, x);
        content = sv_trim_left(content);
    }

    FILE* in = fopen(data_file_path, "rb");
    if (in == NULL) {
        fprintf(stderr, "ERROR: could not read file %s:\n", data_file_path);
        return 1;
    }
    matrix t = matrix_load(in);
    fclose(in);

    NN_ASSERT(arch.count > 1);
    size_t ins_sz = arch.items[0];
    size_t outs_sz = arch.items[arch.count - 1];
    NN_ASSERT(t.cols == ins_sz + outs_sz);

    matrix ti = {
        .rows = t.rows,
        .cols = ins_sz,
        .stride = t.stride,
        .elements = &MATRIX_AT(t, 0, 0),
    };

    matrix to = {
        .rows = t.rows,
        .cols = outs_sz,
        .stride = t.stride,
        .elements = &MATRIX_AT(t, 0, ins_sz),
    };

    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch.items, arch.count);
    nn_randomise(nn, -1, 1);
    NN_DISPLAY(nn);

    float rate = 1;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gui");
    // SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);

    Cost_Plot plot = {0};

    size_t epochs = 0;
    size_t max_epoch = 10000;
    size_t epochs_per_frame = 500;
    bool paused = false;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epochs = 0;
            nn_randomise(nn, -1, 1);
            plot.count = 0;
        }
        for (size_t i = 0; i < epochs_per_frame && !paused && epochs < max_epoch; i++) {
            if (epochs < max_epoch) {
                nn_backprop(nn, &g, ti, to);
                nn_learn(nn, g, rate);
                epochs++;
                float c = nn_cost(nn, ti, to);
                da_append(&plot, c);
                printf("epoch: %zu: cost = %f\n", epochs, c);
            }
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            int rw = w / 2;
            int rh = h * 2 / 3;
            int rx = 0;
            int ry = h / 2 - rh / 2;

            plot_cost(plot, rx, ry, rw, rh);
            rx += rw;

            nn_render_raylib(nn, rx, ry, rw, rh);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu / %zu, Rate = %f, Cost = %f", epochs, max_epoch, rate, nn_cost(nn, ti, to));
            DrawText(buffer, 0, 0, h * 0.04, WHITE);
        }
        EndDrawing();

        // if (i == 10000) CloseWindow();
    }

    #if 0
    size_t n = 16;

    size_t fails = 0;
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < n; y++) {
            size_t z = x + y;
            for (size_t j = 0; j < 4; j++) {
                MATRIX_AT(NN_INPUT(nn), 0, j) = (x>>j)&1;
                MATRIX_AT(NN_INPUT(nn), 0, j + 4) = (y>>j)&1;
            }
            nn_forward(nn);
            if (MATRIX_AT(NN_OUTPUT(nn), 0, 4) > 0.5f) {
                if (z < n) {
                    printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
                    fails += 1;
                }
            } 
            else {
                size_t a = 0;
                for (size_t j = 0; j < 4; j++) {
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
    #endif
    #if 0
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y = MATRIX_AT(NN_OUTPUT(nn), 0, 0);
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
    #endif

    return 0;
}