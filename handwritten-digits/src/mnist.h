#pragma once

#include <stdint.h>

typedef struct {
    int num_images;
    int rows;
    int cols;
    uint8_t *images;
    uint8_t *labels;
} Dataset;

Dataset mnist_load(
    const char* image_path,
    const char* label_path
);

void mnist_free(Dataset *d);