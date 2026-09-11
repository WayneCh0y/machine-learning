#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dataloader.h"

DataLoader dataloader_create(const Dataset *dataset, int batch_size) {
    DataLoader dataloader = { 0 };
    
    int *indices = malloc((size_t)dataset->num_images * sizeof(int));
    if (!indices) {
        fprintf(stderr, "out of memory \n");
        exit(1);
    }
    for (int i = 0; i < dataset->num_images; i++) {
        // During instantiation, we map identity.
        indices[i] = i;
    }

    dataloader.cursor = 0;
    dataloader.data = dataset;
    dataloader.batch_size = batch_size;
    dataloader.indices = indices;

    printf("Dataloader successfully created!\n");
    printf("Batch size is %d", dataloader.batch_size);
    return dataloader;
}

void dataloader_free(DataLoader *dataloader) {
    // We free everything we allocated, and reset the dataloader object.
    // Here, I used (*object).field, but actually '->' should be better.
    // Oh wells, I am more comfortable with this notation.
    free((*dataloader).indices);
    (*dataloader).indices = NULL;
    (*dataloader).data = NULL;
    (*dataloader).batch_size = 0;
    (*dataloader).cursor = 0;
}

void dataloader_reset(DataLoader *dataloader) {
    // For a new epoch, we need to reshuffle, 
    // and reset the cursor to start from the beginning.

    // The easy one, reset the cursor to zero.
    (*dataloader).cursor = 0;

    // The hard one, generate a random permutation for indices.
    int n = (*(*dataloader).data).num_images;

    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i + 1);

        int tmp = (*dataloader).indices[i];
        (*dataloader).indices[i] = (*dataloader).indices[j];
        (*dataloader).indices[j] = tmp;
    }

    return;
}

int dataloader_next(DataLoader *dataloader, Batch *out) {
    // Gets the next batch of data. 
    // Before that, we need to know the current position we are at.
    const Dataset *data = (*dataloader).data;
    int remaining = (*data).num_images - (*dataloader).cursor;

    // If we have nothing remaining, the epoch has ended.
    if (remaining <= 0) {
        return 0;
    }

    // The total number of images are not guarenteed to be divisible by the batch size.
    // Therefore, the size of the batch could be lesser for the last batch.
    int n = (*dataloader).batch_size;
    if (remaining < n) {
        n = remaining;
    }

    int pixels = (*data).rows * (*data).cols;

    for (int b = 0; b < n; b++) {
        // Put in the image one at a time.
        // Get the current "shuffled" index of interest.
        int idx = (*dataloader).indices[(*dataloader).cursor + b];

        // We get the starting point of the image of interest, at the index.
        // Do some arithmetic, offset by num_pixels per image * idx of interest.
        const uint8_t *src = (*data).images + ((size_t)idx * pixels);
        // Starting point of this image at the output.
        // This is dependent on the batch number now.
        float *dst = (*out).images + ((size_t)b * pixels);

        // Normalize to [0,1]
        for (int p = 0; p < pixels; p++) {
            dst[p] = src[p] / 255.0f;
        }

        (*out).labels[b] = (*data).labels[idx];
    }

    (*out).count = n;
    (*dataloader).cursor += n;

    return 1;
}

Batch batch_create(int batch_size, int rows, int cols) {
    Batch batch = { 0 };
    
    // We get the size of the array we want.
    // This is the whole batch information size.
    int size = batch_size * rows * cols;

    float *images = malloc(size * sizeof(float));
    uint8_t *labels = malloc(batch_size * sizeof(uint8_t));

    if (!images || !labels) {
        fprintf(stderr, "out of memory \n");
        exit(1);
    }

    batch.count = 0;
    batch.images = images;
    batch.labels = labels;

    return batch;
}

void batch_free(Batch *batch) {
    free((*batch).images);
    free((*batch).labels);
    (*batch).images = NULL;
    (*batch).labels = NULL;
    (*batch).count = 0;
}
