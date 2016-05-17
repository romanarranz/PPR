#include <iostream>
#include <stdlib.h>
#include <math.h>
using std::cout;
using std::cerr;
using std::endl;

#include "cuda_runtime.h"

#include "sumaMatrices.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

int main(){

    // Get Device Information
    int devID = 0;
    CUDA_CHECK(cudaSetDevice(devID));
    CUDA_CHECK(cudaGetDevice(&devID));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    if (deviceProp.computeMode == cudaComputeModeProhibited){
        cerr << "Error: La GPU no permite realizar computo ahora mismo, las hebras no pueden usar ::cudaSetDevice()." << endl;
        exit(EXIT_SUCCESS);
    }
    else{
        cout << "GPU Device " << devID << ": \"" << deviceProp.name << "\" with compute capability " << deviceProp.major << "." << deviceProp.minor << endl << endl;
    }

    // Use a larger block size for Fermi and above
    int blockS = (deviceProp.major < 2) ? 16 : 32; // si deviceProp.major < 2 => blockSize = 16;  else blockSize = 32;
    int N = 5 * 2 * blockS;

    dim3 blockSize(blockS, blockS);
    int numBloques = ceil( (float) N / blockSize.x);
    int numThreadsBloque = ceil ( (float) N / blockSize.y);
    dim3 numBlocks (numBloques, numThreadsBloque);

    // CPU variables
    float * h_A = NULL;
    float * h_B = NULL;
    float * h_C = NULL;
    initMatrixes(&h_A, &h_B, &h_C, N);

    // Calc
    matAddGPU(h_A, h_B, h_C, N, numBloques, numThreadsBloque);

    cout << "CPU: Mosrando resultados..." << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C[N-1] = " << h_C[N-1] << endl;

    // Liberando memoria de CPU
    free(h_A);
    free(h_B);
    free(h_C);
}
