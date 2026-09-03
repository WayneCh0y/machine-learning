#!/bin/bash
set -e

BASE=https://storage.googleapis.com/cvdf-datasets/mnist

mkdir -p data && cd data

for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
        t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [ -f "$f" ] && continue
    curl -fLO "$BASE/$f.gz"
    gunzip "$f.gz"
done

ls -l