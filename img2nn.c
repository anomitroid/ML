#include <assert.h>
#include <time.h>
#include <float.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GUI
#include "nn.h"

char *args_shift(int *argc, char ***argv) {
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

int main(int argc, char** argv) {
    srand(time(0));
    
    const char* program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <input.png\n>", program);
        fprintf(stderr, "ERROR: no input file is provided\n");
        return 1;
    }

    const char* img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, img_comp;
    uint8_t* img_pixels = (uint8_t*) stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);

    if (img_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
        return 1;
    }

    if (img_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits. Only 8 bit grayscale images are supportefd\n", img_file_path, img_comp * 8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp * 8);


    matrix t = matrix_alloc(img_width * img_height, 3, 3);

    for (int y = 0; y < img_height; y++) {
        for (int x = 0; x < img_width; x++) {
            size_t i = y * img_width + x;
            MATRIX_AT(t, i, 0) = (float) x / (img_width - 1);
            MATRIX_AT(t, i, 1) = (float) y / (img_height - 1);
            MATRIX_AT(t, i, 2) = img_pixels[i] / 255.f;
        }
    }

    matrix ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .elements = &MATRIX_AT(t, 0, 0),
    };

    matrix to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .elements = &MATRIX_AT(t, 0, ti.cols),
    };

    // MATRIX_DISPLAY(ti);
    // MATRIX_DISPLAY(to);

    size_t arch[] = {2, 7, 5, 1};
    NN nn = nn_alloc(arch, ARRAY_SIZE(arch));
    NN g = nn_alloc(arch, ARRAY_SIZE(arch));
    // NN g = nn_alloc(arch.items, arch.count);
    nn_randomise(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16 * WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9 * WINDOW_FACTOR);

    
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gui");
    // SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);
    
    Plot plot = {0};

    Image preview_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    Image original_image = GenImageColor(img_width, img_height, BLACK);
    for (size_t y = 0; y < (size_t) img_height; y++) {
        for (size_t x = 0; x < (size_t) img_width; x++) {
            uint8_t pixel = img_pixels[y * img_width + x];
            ImageDrawPixel(&original_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
    Texture2D original_texture = LoadTextureFromImage(original_image);
    
    size_t epochs = 0;
    size_t max_epoch = 100 * 1000;
    size_t epochs_per_frame = 100;
    float rate = 1.0f;
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

            int rw = w / 3;
            int rh = h * 2 / 3;
            int rx = 0;
            int ry = h / 2 - rh / 2;

            gui_plot(plot, rx, ry, rw, rh);
            rx += rw;
            gui_render_nn(nn, rx, ry, rw, rh);
            rx += rw;
            
            float scale = 20;

            for (size_t y = 0; y < (size_t) img_height; y++) {
                for (size_t x = 0; x < (size_t) img_width; x++) {
                    MATRIX_AT(NN_INPUT(nn), 0, 0) = (float) x / (img_width - 1);
                    MATRIX_AT(NN_INPUT(nn), 0, 1) = (float) y / (img_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                    ImageDrawPixel(&preview_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            UpdateTexture(preview_texture, preview_image.data);
            DrawTextureEx(preview_texture, CLITERAL(Vector2) { rx, ry }, 0, scale, WHITE);
            
            DrawTextureEx(original_texture, CLITERAL(Vector2) { rx, ry + img_height * scale }, 0, scale, WHITE);
            

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu / %zu, Rate = %f, Cost = %f", epochs, max_epoch, rate, nn_cost(nn, ti, to));
            DrawText(buffer, 0, 0, h * 0.04, WHITE);
        }
        EndDrawing();

        // if (i == 10000) CloseWindow();
    }

    for (size_t y = 0; y < (size_t) img_height; y++) {
        for (size_t x = 0; x < (size_t) img_width; x++) {
            uint8_t pixel = img_pixels[y * img_width + x]; 
            if (pixel) printf("%3u ", pixel);
            else printf("   ");
        }
        printf("\n");
    }

    for (size_t y = 0; y < (size_t) img_height; y++) {
        for (size_t x = 0; x < (size_t) img_width; x++) {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = (float) x / (img_width - 1);
            MATRIX_AT(NN_INPUT(nn), 0, 1) = (float) y / (img_width - 1);
            nn_forward(nn);
            uint8_t pixel = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
            if (pixel) printf("%3u ", pixel);
            else printf("   ");
        }
        printf("\n");
    }

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t* out_pixels = malloc(sizeof(*out_pixels) * out_height * out_width);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < (size_t) out_height; y++) {
        for (size_t x = 0; x < (size_t) out_width; x++) {
            MATRIX_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
            MATRIX_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MATRIX_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
            out_pixels[y * out_width + x] = pixel;
        }
    }   

    const char* out_file_path = "upscaled.png";

    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img_file_path);

    return 0;
}