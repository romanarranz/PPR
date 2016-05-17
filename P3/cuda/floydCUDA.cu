#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"
#include "floyd.h"

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

__global__ void floyd1DKernel(int * M, const int nverts, const int k){
    short j = blockIdx.x * blockDim.x + threadIdx.x;    // indice filas
    short i = blockIdx.y;                               // indice columnas
    short tid = (i * nverts) + j;

    if(i!=j && i!=k && j!=k){
        if (i!=j && i!=k && j!=k) {
            short jk = (j*nverts) + k;
            short ki = (k*nverts) + i;
            short ij = (j*nverts) + i;
            int aux = M[jk]+M[ki];

            int vikj = min(aux, M[ij]);
            M[tid] = vikj;
        }
    }
}

// Kernel to update the Matrix at k-th iteration
__global__ void floyd2DKernel(int * M, const int nverts, const int k){
    short j = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
    short i = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
    short tid = (i * nverts) + j;

    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k) {
            short jk = (j*nverts) + k;
            short ki = (k*nverts) + i;
            short ij = (j*nverts) + i;
            int aux = M[jk]+M[ki];

            int vikj = min(aux, M[ij]);
            M[tid] = vikj;
        }
    }
}

void floyd1DGPU(int *h_M, Graph g, int N, int numBloques, int numThreadsBloque){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_M, memSize));

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));

    cout << "GPU: Calculando..." << endl;
    for(int k = 0; k < N; k++){
        floyd1DKernel<<< numBloques, numThreadsBloque >>> (d_M, N, k);
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));
}

void floyd2DGPU(int *h_M, Graph g, int N, int numBloques, int numThreadsBloque){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_M, memSize));

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));

    cout << "GPU: Calculando..." << endl;
    for(int k = 0; k < N; k++){
        dim3 threadsPerBlock(numThreadsBloque,numThreadsBloque);
        dim3 numBlocks (numBloques, numThreadsBloque);
        floyd2DKernel<<< numBlocks, threadsPerBlock >>> (d_M, N, k);
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));
}
