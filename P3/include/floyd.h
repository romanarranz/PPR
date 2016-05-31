#ifndef FLOYD_H
#define FLOYD_H

#include "cuda_runtime.h"

void floyd1DGPU(int *M, int N, int numBloques, int numThreadsBloque);
void floyd1DSharedMGPU(int *M, int N, int numBloques, int numThreadsBloque);

void floyd2DGPU(int *M, int N, dim3 numBlocks, dim3 threadsPerBlock);
void floyd2DSharedMGPU(int *M, int N, dim3 numBlocks, dim3 threadsPerBlock);

#endif
