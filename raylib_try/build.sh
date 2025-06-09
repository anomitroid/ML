#!/bin/sh

set -xe

clang `pkg-config --cflags raylib` -o main main.c `pkg-config --libs raylib` -lm -lglfw -ldl -lpthread