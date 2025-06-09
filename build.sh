#/!bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm -lglfw -ldl -lpthread"

# clang -Wall -Wextra -o nn nn.c -lm
# clang $CFLAGS -o nn1 nn1.c $LIBS
clang $CFLAGS -o gui gui.c $LIBS
# clang $CFLAGS -o adder_gen adder_gen.c $LIBS
# clang $CFLAGS -o xor_gen xor_gen.c $LIBS