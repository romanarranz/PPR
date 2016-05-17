#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "sumaMatrices.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

// Device Code: Kernel 2D
__global__ void matAdd(float *A, float *B, float *C, int N){
    // Las matrices se recorren con la ordenacion de Fortran
    int j = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
    int i = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
    int tid = (i * N) + j;

    if(i < N && j < N)
        C[tid] = A[tid] + B[tid];
}

// http://stackoverflow.com/questions/1398307/how-can-i-allocate-memory-and-return-it-via-a-pointer-parameter-to-the-calling
void initMatrixes(float * *h_A, float * *h_B, float * *h_C, int N){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(float);
    *h_A = (float *) malloc(memSize);
    *h_B = (float *) malloc(memSize);
    *h_C = (float *) malloc(memSize);

    cout << "CPU: Inicializando los vectores..." << endl;
    for(int i = 0; i<N; i++){
        int row = N*i;
        for(int j = 0; j<N; j++){
            *h_A[row+j] = i+1;
            *h_B[row+j] = (i+1)*10;
            *h_C[row+j] = 0;
        }
    }
    int lastIndex = (N * N)-1;
    cout << "h_A[0] = " << h_A[0] << " ... h_A[N*N-1] = " << h_A[lastIndex] << endl;
    cout << "h_B[0] = " << h_B[0] << " ... h_B[N*N-1] = " << h_B[lastIndex] << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C[N*N-1] = " << h_C[lastIndex] << endl;
}

void matAddGPU(float *h_A, float *h_B, float *h_C, int N, int numBloques, int numThreadsBloque){
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(float);

    // GPU variables
    float * d_A = NULL;
    float * d_B = NULL;
    float * d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, memSize));
    CUDA_CHECK(cudaMalloc((void **)&d_B, memSize));
    CUDA_CHECK(cudaMalloc((void **)&d_C, memSize));

    if(h_A == NULL){
      cout << "LA HEMOS LIAO" << endl;
    }
    cout << "CPU: Copiando las matrices de la CPU RAM a la GPU DRAM..." << endl;
    cudaError_t error;
    error = cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
      cout << "cudaMemcpy (d_A,h_A) returned error '"<< cudaGetErrorString(error) << "' (code "<< error << "), line("<< __LINE__ << ")" << endl;
      exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice));

    // Llamada a CUDA Kernel con N datos, usando 256 threads y bloque 1D, recordamos que los wraps de hebras se cogen de 32 hebras en 32
    // callback <<blocks_per_grid, thread_per_block>> (params);
    cout << "GPU: Calculando..." << endl;
    matAdd<<<numBloques, numThreadsBloque>>> (d_A, d_B, d_C, N);

    cout << "CPU: Copiando los resultados de la GPU DRAM a la CPU RAM..." << endl;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, memSize, cudaMemcpyDeviceToHost));

    cout << "CPU: Liberando los datos de la GPU DRAM" << endl;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaDeviceReset());
}
