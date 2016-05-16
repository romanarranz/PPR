#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "sumaVectores.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

// Device Code: Kernel 1D
__global__ void vecAdd(float *A, float *B, float *C, int N){
    /*
        Variables predefinidas
        =====================
        uint3 blockIdx: coordenadas de un bloque en la malla. (blockIdx.x, blockIdx.y, blockIdx.z).
        dim3 blockDim: dimensiones del bloque.
        dim3 gridDim: dimensiones de la malla.
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N)
        C[tid] = A[tid] + B[tid];
}

void initVectors(float *h_A, float *h_B, float *h_C, int N){
    cout << "CPU: Inicializando los vectores..." << endl;
    for(int i = 0; i<N; i++){
        h_A[i] = i+1;
        h_B[i] = (i+1)*10;
        h_C[i] = 0;
    }
    cout << "h_A[0] = " << h_A[0] << " ... h_A[N-1] = " << h_A[N-1] << endl;
    cout << "h_B[0] = " << h_B[0] << " ... h_B[N-1] = " << h_B[N-1] << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C[N-1] = " << h_C[N-1] << endl;
}

void computeGPU(float *h_A, float *h_B, float *h_C, int N, int numBloques, int numThreadsBloque){

    int size = N * sizeof(float);

    // GPU variables
    float * d_A = NULL;
    float * d_B = NULL;
    float * d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    cout << "CPU: Copiando los vectores de la CPU RAM a la GPU DRAM..." << endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Llamada a CUDA Kernel con N datos, usando 256 threads y bloque 1D, recordamos que los wraps de hebras se cogen de 32 hebras en 32
    // callback <<blocks_per_grid, thread_per_block>> (params);
    cout << "GPU: Calculando..." << endl;
    vecAdd<<<numBloques, numThreadsBloque>>> (d_A, d_B, d_C, N);

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    cout << "CPU: Liberando los datos de la GPU DRAM" << endl;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
