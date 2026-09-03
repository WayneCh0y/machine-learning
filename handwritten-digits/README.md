# Handwritten Digits in C

For this project, I shall attempt the following:

1. Write a CNN in C.
2. Write cuda kernels for both training and inference.

We shall use the MNIST handwritten digit dataset.

## Obtaining Dataset

To obtain the dataset, simply run the [`get_mnist.sh`](/handwritten-digits/get_mnist.sh) script that I have written. simply run (on macOS):

```
chmod +x get_mnist.sh
./get_mnist.sh
```