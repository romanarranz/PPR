#include <iostream>
#include <stdio.h>
using std::cout;
using std::cerr;
using std::endl;

#include "floyd.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// solo defino una vez el area de memoria compartida de forma dinamica, para que por cada llamada CUDA vuelque aqui el contenido
extern __shared__ int smem[];

__global__ void floyd1DSharedMKernel(int * M, const int nverts, const int k, const int blockSize){
    // <== INICIALIZACION
    // ====================================>
    int li = threadIdx.x;                             // indice local en el vector de memoria compartida (shared)
    int ii = blockIdx.x * blockDim.x + threadIdx.x;   // coincide con ij, indice filas en el vector de memoria global
    int i = ii/nverts;
    int j = ii - (i*nverts);

    // <== PREPARAR DATOS
    // ====================================>
    // vectores de memoria compartida
    int *s_rowI = (int *) &smem; //a is manually set at the beginning of shared
    int *s_rowK = (int *) &s_rowI[blockSize/2]; //b is manually set at the end of a
    __shared__ int s_ik[2];

    // cargar los datos al vector de memoria compartida
    int kj = (k*nverts) + j;
    s_rowI[li] = M[ii];
    s_rowK[li] = M[kj];

    // si es la primera hebra del bloque, guardamos la k de esa fila M(i,k)
    if (li == 0)
        s_ik[0] = M[i * nverts + k];

    // si es la ultima hebra del bloque, guardamos la k de la siguiente fila M(i+1,k)
    if(li == blockSize - 1)
        s_ik[1] = M[i * nverts + k];

    __syncthreads();

    // <== CALCULO
    // ====================================>
    // fila de la primera hebra del bloque
    int rowFirstBTid = floor((double) (blockIdx.x * blockDim.x) / nverts);

    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k){
            // thread en misma fila, calculamos con la k de esa fila
            if (rowFirstBTid == i)
                M[ii] = min(s_ik[0] + s_rowK[li], s_rowI[li]);
            // thread en distinta fila, calculamos con la k de la sig fila
            else
                M[ii] = min(s_ik[1] + s_rowK[li], s_rowI[li]);
        }
    }
}

__global__ void floyd2DSharedMKernel(int * M, const int nverts, const int k, const int blockSize){
    // <== INICIALIZACION
    // ====================================>
    int li = threadIdx.y;
    int lj = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ij = (i*nverts) + j;

    // <== PREPARAR DATOS
    // ====================================>
    // vectores de memoria compartida
    int * s_rowK = (int *) &smem;                    // filak del bloque
    int * s_colK = (int *) &s_rowK[blockSize/2];     // columnak del bloque

    // cargar los datos al vector de memoria compartida
    int ik = (k*nverts) + j;
    int kj = (i*nverts) + k;

    if (li == 0)
      s_rowK[lj] = M[ik];

    if (lj == 0)
      s_colK[li] = M[kj];

    __syncthreads();

    // <== CALCULO
    // ====================================>
    if(i < nverts && j < nverts){
        if (i!=j && i!=k && j!=k){
            M[ij] = min(s_rowK[lj] + s_colK[li], M[ij]);
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
    int blockSize = numThreadsBloque * 2; // 2 * blockSize
    for(int k = 0; k < N; k++){
        floyd1DSharedMKernel<<< nblocks, threadsPerBlock, blockSize * sizeof(int) >>> (d_M, N, k, blockSize );
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}

void floyd2DSharedMGPU(int *h_M, int N, dim3 numBlocks, dim3 threadsPerBlock){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(int);

    // GPU variables
    int * d_M = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_M, memSize));

    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice));

    cout << "GPU: Calculando..." << endl;
    int blockSize = threadsPerBlock.x * 2;
    for(int k = 0; k < N; k++){
        floyd2DSharedMKernel<<< numBlocks, threadsPerBlock, blockSize * sizeof(int) >>> (d_M, N, k, blockSize );
    }

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_M, d_M, memSize, cudaMemcpyDeviceToHost));

    // Flush all profile data before the application exits
    cudaDeviceReset();
}
