#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra -I./headers/"
LIBS="-lm"

clang $CFLAGS `pkg-config --cflags raylib` -o xor xor.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o adder adder.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o gui gui.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o img2nn img2nn.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread