#include <stdio.h>
#include "raylib.h"

int main(void) {
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "hagu");

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        {
            ClearBackground(RAYWHITE);
            DrawCircle(screenWidth / 2, screenHeight / 2, 100, RED);
            // DrawText("hagu", screenWidth / 2, screenHeight / 2, 69, LIGHTGRAY);
        }
        EndDrawing();
    }

    CloseWindow();

    return 0;
}