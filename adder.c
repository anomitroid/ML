#include <math.h>
#include <time.h>
#include <string.h>
#include <raylib.h>

#define NN_IMPLEMENTATION
#define NN_ENABLE_GUI
#include "nn.h"

#define BITS 3

#ifndef CLAMP
    #define CLAMP(v, lo, hi)  ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#endif

static void draw_weight_heatmap(NN nn, int li,
                                                                float rx, float ry, float rw, float rh) {
        matrix W = nn.weights[li];
        size_t rows = W.rows;
        size_t cols = W.cols;
        float cellW = rw / cols;
        float cellH = rh / rows;
        float wmin = INFINITY, wmax = -INFINITY;
        for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                        float w = W.elements[r * W.stride + c];
                        wmin = fminf(wmin, w);
                        wmax = fmaxf(wmax, w);
                }
        }
        float range = (wmax - wmin);
        if (range < 1e-6f) range = 1.0f;
        for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                        float w = W.elements[r * W.stride + c];
                        float t = (w - wmin) / range;
                        unsigned char v = (unsigned char)(t * 255);
                        Color col = (Color){ v, v, v, 0xFF };
                        Rectangle cell = { rx + c*cellW, ry + r*cellH, cellW+1, cellH+1 };
                        DrawRectangleRec(cell, col);
                }
        }
        DrawTextEx(GetFontDefault(), TextFormat("W%d [%zu×%zu]", li, rows, cols),
                             (Vector2){rx, ry - 16}, 14, 0, WHITE);
}

void verify_nn_adder(Font font, NN nn, float rx, float ry, float rw, float rh) {
        const size_t n     = (1 << BITS);
        const size_t total = n * n;
        float aspect  = rw/rh;
        int   columns = (int)ceilf(sqrtf(total * aspect));
                    columns = CLAMP(columns, 1, total);
        int   rows    = (total + columns - 1)/columns;
        float cellW = rw/columns;
        float cellH = rh/rows;
        const float PAD      = 2.0f;
        float       s_small  = fminf((cellH - PAD)/2.0f, (cellW*0.9f)/6.0f);
                                s_small  = CLAMP(s_small, 6.0f, 16.0f);
        size_t idx = 0;
        for (size_t x = 0; x < n; x++) {
                for (size_t y = 0; y < n; y++, idx++) {
                        char add[16];
                        snprintf(add, sizeof add, "%zu+%zu", x, y);
                        Vector2 m = MeasureTextEx(font, add, s_small, 0);
                        float bx = rx + (idx % columns)*cellW + (cellW - m.x)/2;
                        float by = ry + (idx / columns)*cellH + (cellH - (2*s_small+PAD))/2;
                        DrawTextEx(font, add, (Vector2){bx, by}, s_small, 0, WHITE);
                }
        }
        Vector2 mpos = GetMousePosition();
        if (mpos.x >= rx && mpos.x < rx+rw && mpos.y >= ry && mpos.y < ry+rh) {
                int hcol = (int)floorf((mpos.x - rx) / cellW);
                int hrow = (int)floorf((mpos.y - ry) / cellH);
                size_t hidx = (size_t)hrow*columns + hcol;
                if (hidx < total) {
                        size_t x = hidx / n;
                        size_t y = hidx % n;
                        size_t sum = x + y;
                        for (size_t j = 0; j < BITS; j++) {
                                MATRIX_AT(NN_INPUT(nn), 0, j)        = (x >> j) & 1;
                                MATRIX_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
                        }
                        nn_forward(nn);
                        size_t approx = 0;
                        for (size_t j = 0; j < BITS; j++) {
                                approx |= ((MATRIX_AT(NN_OUTPUT(nn), 0, j) > 0.5f) << j);
                        }
                        bool overflow = MATRIX_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f;
                        char line1[32], line2[32];
                        snprintf(line1, sizeof line1, "%zu+%zu=%zu", x, y, sum);
                        snprintf(line2, sizeof line2, "NN:%2zu%s", approx, overflow?"O":"");
                        const float s_big   = 24.0f;
                        const float MARGIN  = 8.0f;
                        Vector2 m1 = MeasureTextEx(font, line1, s_big, 0);
                        Vector2 m2 = MeasureTextEx(font, line2, s_big, 0);
                        float boxW = fmaxf(m1.x, m2.x) + MARGIN*2;
                        float boxH = m1.y + m2.y + MARGIN*3;
                        float bx = mpos.x + 10;
                        float by = mpos.y + 10;
                        if (bx + boxW > rx+rw) bx = rx+rw - boxW - 5;
                        if (by + boxH > ry+rh) by = ry+rh - boxH - 5;
                        DrawRectangleRounded((Rectangle){bx,by,boxW,boxH}, 0.1f, 8, Fade(BLACK,0.8f));
                        DrawTextEx(font, line1, (Vector2){bx+MARGIN, by+MARGIN}, s_big, 0, GOLD);
                        DrawTextEx(font, line2, (Vector2){bx+MARGIN, by+MARGIN + m1.y + MARGIN/2}, s_big, 0, GOLD);
                }
        }
}

int main(void) {
        srand((unsigned)time(NULL));
        size_t n       = (1 << BITS);
        size_t samples = n * n;
        matrix ti = matrix_alloc(samples, 2 * BITS, 2 * BITS);
        matrix to = matrix_alloc(samples, BITS + 1, BITS + 1);
        for (size_t i = 0; i < samples; i++) {
                size_t x = i / n, y = i % n, sum = x + y;
                for (size_t j = 0; j < BITS; j++) {
                        ti.elements[i * ti.stride + j]        = (x >> j) & 1;
                        ti.elements[i * ti.stride + j + BITS] = (y >> j) & 1;
                        to.elements[i * to.stride + j]        = (sum >> j) & 1;
                }
                to.elements[i * to.stride + BITS] = (sum >= n);
        }
        size_t arch[] = { 2 * BITS, 4 * BITS, BITS + 1 };
        NN nn   = nn_alloc(arch, ARRAY_SIZE(arch));
        NN grad = nn_alloc(arch, ARRAY_SIZE(arch));
        nn_randomise(nn, -1.0f, 1.0f);
        const int WIN_F = 80;
        InitWindow(16*WIN_F, 9*WIN_F, "NN Adder");
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        SetTargetFPS(60);
        Font font = LoadFontEx("./fonts/iosevka-regular.ttf", 72, NULL, 0);
        SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
        Plot plot = { 0 };
        size_t epoch = 0, max_epoch = 100000, eps = 100;
        float rate = 0.1f;
        bool paused = false;
        while (!WindowShouldClose()) {
                if (IsKeyPressed(KEY_SPACE)) paused = !paused;
                if (IsKeyPressed(KEY_R)) {
                        epoch = 0;
                        nn_randomise(nn, -1.0f, 1.0f);
                        plot.count = 0;
                }
                for (size_t i = 0; i < eps && !paused && epoch < max_epoch; i++) {
                        nn_backprop(nn, &grad, ti, to);
                        nn_learn(nn, grad, rate);
                        da_append(&plot, nn_cost(nn, ti, to));
                        epoch++;
                }
                BeginDrawing();
                ClearBackground((Color){0x18,0x18,0x18,0xFF});
                int W = GetRenderWidth(), H = GetRenderHeight();
                int cellW = W/3, cellH = H*2/3, offY = H/2 - cellH/2;
                gui_plot(plot, 0, offY, cellW, cellH);
                gui_render_nn(nn, cellW, offY, cellW, cellH);
                verify_nn_adder(font, nn, 2*cellW, offY, cellW, cellH);
                draw_weight_heatmap(nn, 0, 2*cellW, offY + cellH + 20,
                                                        cellW, H - (offY + cellH + 20) - 20);
                char st[64];
                snprintf(st, sizeof st, "Epoch %zu/%zu  Rate %.3f  Cost %.4f",
                                 epoch, max_epoch, rate, nn_cost(nn, ti, to));
                DrawTextEx(font, st, (Vector2){10,10}, H*0.04f, 0, WHITE);
                EndDrawing();
        }
        nn_free(&nn);
        nn_free(&grad);
        matrix_free(&ti);
        matrix_free(&to);
        CloseWindow();
        return 0;
}
