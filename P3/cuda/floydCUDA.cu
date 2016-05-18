#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "floyd.h"

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

// Kernel to update the Matrix at k-th iteration
__global__ void floyd1DKernel(int * M, const int nverts, const int k){
    int ii = blockIdx.x * blockDim.x + threadIdx.x;    // indice filas, coincide con ij
    int i = ii/nverts;
    int j = ii - i * nverts;

    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k) {
            int ik = (i*nverts) + k;
            int kj = (k*nverts) + j;
            M[ii] = min(M[ik] + M[kj], M[ii]);
        }
    }
}

// Kernel to update the Matrix at k-th iteration
__global__ void floyd2DKernel(int * M, const int nverts, const int k){
    int jj = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
    int ii = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
    int tid = (ii * nverts) + jj;
    int i = tid/nverts;
    int j = tid - i * nverts;
    //printf ("Fila %u, Columna %u => Thread id %d.\n", i, j, tid);

    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k) {
            int ik = (i*nverts) + k;
            int kj = (k*nverts) + j;
            int ij = (i*nverts) + j;
            int aux = M[ik]+M[kj];

            int vikj = min(aux, M[ij]);
            M[ij] = vikj;
        }
    }
}

void floyd1DGPU(int *h_M, int N, int numBloques, int numThreadsBloque){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_M, memSize));

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));

    cout << "GPU: Calculando..." << endl;
    dim3 nblocks(numBloques);
    dim3 threadsPerBlock(numThreadsBloque, 1);
    for(int k = 0; k < N; k++){
        floyd1DKernel<<< nblocks, threadsPerBlock >>> (d_M, N, k);
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}

void floyd2DGPU(int *h_M, int N, dim3 numBlocks, dim3 threadsPerBlock){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_M, memSize));

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));

    cout << "GPU: Calculando..." << endl;
    for(int k = 0; k < N; k++){
        floyd2DKernel<<< numBlocks, threadsPerBlock >>> (d_M, N, k);
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}
