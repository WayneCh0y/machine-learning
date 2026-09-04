#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist.h"

int main(void) {
    /**
     * This is the entrypoint of our training algorithm :)
     * We first load the dataset (training set), we shall batch it as well, and normalize.
     * During dataloading, we shall shuffle (or perhaps stratify the batches), to ensure every batch has an image from every class.
     * We pass it through the convolution and FC layers, then calculate loss, and backpropagate.
     */

    srand((unsigned)time(NULL));

    Dataset train = mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    mnist_free(&train);

    return 0;
}