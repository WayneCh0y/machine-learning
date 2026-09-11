#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist.h"
#include "dataloader.h"

#define BATCH_SIZE 32
#define NUM_EPOCHS 16

int main(void) {
    /**
     * This is the entrypoint of our training algorithm :)
     * We first load the dataset (training set), we shall batch it as well, and normalize.
     * During dataloading, we shall shuffle, to ensure every batch has an image from every class.
     * We pass it through the convolution and FC layers, then calculate loss, and backpropagate.
     */

    srand((unsigned)time(NULL));

    Dataset train = mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    DataLoader train_loader = dataloader_create(&train, BATCH_SIZE);

    // Freeing stuff.
    mnist_free(&train);
    dataloader_free(&train_loader);

    return 0;
}