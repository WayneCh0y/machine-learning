#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include "mnist.h"

/**
 * The first 4 bytes of MNIST file denotes the type.
 * 2051 is for image files.
 * 2049 is for label files.
 * We shall declare them as constants here, so the checking is easier.
 */
#define MNIST_IMAGE 2051
#define MNIST_LABEL 2049

/**
 * For the mnist dataset, we expect a 28 by 28 pixel image.
 */
#define WIDTH 28
#define HEIGHT 28

/**
 * The files are stored in big-endian.
 * However OS is a little-endian machine.
 * Reverse the order of bytes when reading a file.
 */
static uint32_t read_big_endian_u32(FILE *f) {
    // 32 bits is 4 bytes. We declare a 4 byte buffer, and fread()
    // Note here, after the fread(), the cursor is moved 32 bits in the file.
    unsigned char buf[4] = { 0 };
    fread(buf, 4, 1, f);

    // Reconstructing the bits in big-endian. 
    return ((uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 | (uint32_t)buf[2] << 8 | (uint32_t)buf[3]);
}

/**
 * We write a helper function to print out the statistics of the dataset.
 */
static void print_dataset_statistics(
    uint32_t image_magic,
    uint32_t image_count,
    uint32_t rows,
    uint32_t cols,
    uint32_t label_magic,
    uint32_t label_count) {
    printf("       | Magic number    Count    Rows    Cols    \n");
    printf("-------------------------------------------------\n");
    printf("Images |     %u        %u     %u      %u\n", image_magic, image_count, rows, cols);
    printf("Labels |     %u        %u\n", label_magic, label_count);

    return;
}

// Thanks claude for this visualization function.
static void mnist_print_ascii(const Dataset *d, int index) {
    const uint8_t *img = d->images + (size_t)index * d->rows * d->cols;
    const char *ramp = " .:-=+*#%@";

    printf("\nSample #%d  |  label: %u\n", index, d->labels[index]);
    for (int r = 0; r < d->rows; r++) {
        for (int c = 0; c < d->cols; c++) {
            char ch = ramp[img[r * d->cols + c] * 9 / 255];
            putchar(ch);
            putchar(ch);
        }
        putchar('\n');
    }
}

Dataset mnist_load(const char* image_path, const char* label_path) {
    // Read the image and label files. The field "rb" is to read binary.
    FILE *f_image = fopen(image_path, "rb");
    FILE *f_label = fopen(label_path, "rb");

    // We handle the case where we cannot open the file.
    if (!f_image || !f_label) {
        fprintf(stderr, "Open failed\n");
        exit(1);
    }

    uint32_t image_magic = read_big_endian_u32(f_image);
    uint32_t image_count = read_big_endian_u32(f_image);
    uint32_t rows = read_big_endian_u32(f_image);
    uint32_t cols = read_big_endian_u32(f_image);

    uint32_t label_magic = read_big_endian_u32(f_label);
    uint32_t label_count = read_big_endian_u32(f_label);

    print_dataset_statistics(image_magic, image_count, rows, cols, label_magic, label_count);

    // Error handling of mismatch counts, mismatch magic numbers.
    if (image_magic != MNIST_IMAGE || label_magic != MNIST_LABEL || image_count != label_count) {
        fprintf(stderr, "Loaded images have a mismatch, exiting...\n");
        exit(1);
    }

    if (rows != HEIGHT || cols != WIDTH) {
        fprintf(stderr, "%s: unexpected dimensions %ux%u (expected (28 by 28))\n", image_path, rows, cols);
        exit(1);
    }
    
    // Now we are ready to write the dataloading, and populate our Dataset struct.
    // We first get the images, as well as the labels.
    size_t image_bytes = (size_t)image_count * rows* cols;
    size_t label_bytes = (size_t)label_count;

    uint8_t* images = malloc(image_bytes);
    uint8_t* labels = malloc(label_bytes);
    if (!images || !labels) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    // From the reading, both cursors are supposedly already at the start of their data.
    size_t n_images = fread(images, 1, image_bytes, f_image);
    if (n_images != image_bytes) {
        fprintf(stderr, "truncated images\n");
        exit(1);
    }

    size_t n_labels = fread(labels, 1, label_bytes, f_label);
    if (n_labels != label_bytes) {
        fprintf(stderr, "truncated labels\n");
        exit(1);
    }
    
    fclose(f_image);
    fclose(f_label);

    Dataset d;
    d.num_images = (int)image_count;
    d.rows = (int)rows;
    d.cols = (int)cols;
    d.images = images;
    d.labels = labels;

    printf("\nDataset loaded! ");

    printf("Visualizing a sample...\n");
    mnist_print_ascii(&d, rand() % d.num_images);
    
    return d;
}

void mnist_free(Dataset *d) {
    free(d->images);
    free(d->labels);
    d->images = NULL;
    d->labels = NULL;
    d->num_images = 0;
}