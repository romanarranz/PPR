#include <iostream>
#include <stdio.h>
using std::cout;
using std::cerr;
using std::endl;

#include "floyd.h"

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

// Kernel to update the Matrix at k-th iteration
extern __shared__ int smem[];
__global__ void floyd1DSharedMKernel(int * M, const int nverts, const int k, const int blockSize){
    int li = threadIdx.x;                             // indice local en el vector de memoria compartida (shared)
    int ii = blockIdx.x * blockDim.x + threadIdx.x;   // indice filas en el vector de memoria global
    int i = ii/nverts;
    int j = ii - (i*nverts);

    // vectores de memoria compartida
    int* s_rowI = &smem[0];
    int* s_rowK = &smem[blockSize];

    // cargar los datos al vector de memoria compartida
    // int ij = (i*nverts) + j; = ii asi que no es necesario
    int kj = (k*nverts) + j;
    s_rowI[li] = M[ii];
    s_rowK[li] = M[kj];
    __syncthreads();
    printf("TID = %u \n\tI = %u => \tS_I[%u] = %u \t? M[%u] = %u \n \tK = %u => \tS_K[%u] = %u \t? M[%u] = %u  \n", ii, i, li, s_rowI[li], ii, M[ii], k, li, s_rowK[li], kj, M[kj]);
    
    // realizar el calculo
    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k) {
            int ik = (k*nverts) - (ii + k) - 1;
            int aux = s_rowI[ik] + s_rowK[li];

            int vikj = min(aux, M[li]);
            M[ii] = vikj;
        }
    }
}

void floyd1DSharedMGPU(int * h_M, int N, int numBloques, int numThreadsBloque){
    // Compute Capability             1.x    2.x - 3.x
    // ---------------------------------------------------
    // Threads per block              512    1024
    // Max shared memory (per block)  16KB   48KB

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
    int blockSize = numBloques * 2;
    for(int k = 0; k < N; k++){
        floyd1DSharedMKernel<<< nblocks, threadsPerBlock, blockSize * sizeof(int) >>> (d_M, N, k, blockSize );
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}
