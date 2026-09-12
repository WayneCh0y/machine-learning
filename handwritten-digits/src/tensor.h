#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// For x[n][c][h][w]
#define IDX4(n,c,h,w, C,H,W) \
    ((size_t)(n)*(size_t)(C)*(size_t)(H)*(size_t)(W) \
     + (size_t)(c)*(size_t)(H)*(size_t)(W) \
     + (size_t)(h)*(size_t)(W) \
     + (size_t)(w))

#define IDX2(n,i, D) \
    ((size_t)(n)* (size_t)(D) \
     + (size_t)(i))

#define WIDX(oc,ic,kh,kw, IC,K) \
    ((size_t)(oc)*(size_t)(IC)*(size_t)(K)*(size_t)(K) \
     + (size_t)(ic)*(size_t)(K)*(size_t)(K) \
     + (size_t)(kh)*(size_t)(K) \
     + (size_t)(kw))

static inline float* alloc_float(size_t count) {
    float *p = malloc(count * sizeof(float));
    if (!p) {
        fprintf(stderr, "out of memory \n");
        exit(1);
    }
    return p;
}

static inline int* alloc_int(size_t count) {
    int* p = malloc(count * sizeof(int));
    if (!p) {
        fprintf(stderr, "out of memory \n");
        exit(1);
    }
    return p;
}

static inline void zero_f(float* p, size_t count) {
    memset(p, 0, count*sizeof(float));
}

static inline int out_dim(int in, int k, int pad, int stride) {
    return ((in + 2*pad - k) / stride) + 1;
}