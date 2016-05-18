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
__global__ void floyd1DSharedMKernel(int * M, const int nverts, const int k, const int blockSize){
    short li = threadIdx.x;                             // indice local en el vector de memoria compartida (shared)
    short ii = blockIdx.x * blockDim.x + threadIdx.x;   // indice filas en el vector de memoria global
    short i = ii/nverts;
    short j = ii - (i*nverts);

    // vector de memoria compartida
    extern __shared__ int s_M[];

    // cargar los datos al vector de memoria compartida
    short indexKBlock = blockSize + j;
    short kj = (k*nverts) + j;
    s_M[li] = M[ii];
    s_M[indexKBlock] = M[kj];
    __syncthreads();

    // realizar el calculo
    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k) {
            short ik = (i*nverts) + k;
            int aux = s_M[ik] + s_M[kj];

            int vikj = min(aux, s_M[li]);
            M[ii] = vikj;
        }
    }
}

void floyd1DSharedMGPU(int *h_M, int blockSize, int N, int numBloques, int numThreadsBloque){
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
    short blockS = (2 * blockSize + 2);
    for(int k = 0; k < N; k++){
        floyd1DSharedMKernel<<< numBloques, numThreadsBloque, blockS>>> (d_M, N, k, blockS);
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}
