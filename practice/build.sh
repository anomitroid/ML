#/!bin/sh

set -xe

clang -Wall -Wextra -o adder adder.c -lm
clang -Wall -Wextra -o xor xor.c -lm