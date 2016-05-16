#include <iostream>

using namespace std;

// Kernel 2D
__global__ void matAdd(float *A, float *B, float *C, int N){
    // Las matrices se recorren con la ordenacion de Fortran
    int j = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
    int i = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
    int index = (i * N) + j;

    if(i < N && j < N)
        C[index] = A[index] + B[index];
}

int main(){
    // <== Bloques 2D
    // ==============================>
    dim3 threadsPerBlock (16,16);

    int numBloques = ceil( (float) N / threadsPerBlock.x);
    int numThreadsBloque = ceil ( (float) N / threadsPerBlock.y);
    dim3 numBlocks (numBloques, numThreadsBloque);

    MatAdd <<<numBlocks, threadsPerBlock>>> (A, B, C, N);
}
