#pragma once

#include <stdint.h>
#include <stdio.h>

#include "mnist.h"

typedef struct {
    const Dataset *data;
    int *indices;
    int batch_size;
    int cursor;
} DataLoader;

typedef struct {
    float *images;
    uint8_t *labels;
    int count;
} Batch;

DataLoader dataloader_create(const Dataset *dataset, int batch_size);
void dataloader_reset(DataLoader *dataloader);
int dataloader_next(DataLoader *dataloader, Batch *out);
void dataloader_free(DataLoader *dataloader);

Batch batch_create(int batch_size, int rows, int cols);
void batch_free(Batch *batch);