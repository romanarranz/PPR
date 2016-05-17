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
    int N = 32 * blockS;

    dim3 blockSize(blockS, blockS);
    int numBloques = ceil( (float) N / blockSize.x);
    int numThreadsBloque = ceil ( (float) N / blockSize.y);
    dim3 numBlocks (numBloques, numThreadsBloque);
    cout << "El blockSize es de: " << blockS << endl;
    cout << "El numBloques es de: " << numBloques << endl;
    cout << "El numThreadsBloque es de: " << numThreadsBloque << endl;

    // CPU variables
    float * h_A = NULL;
    float * h_B = NULL;
    float * h_C = NULL;
    unsigned int sizeMatrix = N * N;
    unsigned int memSize = sizeMatrix * sizeof(float);
    h_A = (float *) malloc(memSize);
    h_B = (float *) malloc(memSize);
    h_C = (float *) malloc(memSize);

    initMatrixes(h_A, h_B, h_C, N);

    int lastIndex = sizeMatrix-1;
    cout << "h_A[0] = " << h_A[0] << " ... h_A["<< lastIndex <<"] = " << h_A[lastIndex] << endl;
    cout << "h_B[0] = " << h_B[0] << " ... h_B["<< lastIndex <<"] = " << h_B[lastIndex] << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C["<< lastIndex <<"] = " << h_C[lastIndex] << endl;

    // Calc
    matAddGPU(h_A, h_B, h_C, N, numBloques, numThreadsBloque);

    cout << "CPU: Mostrando resultados..." << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C["<< lastIndex <<"] = " << h_C[lastIndex] << endl;

    // Liberando memoria de CPU
    free(h_A);
    free(h_B);
    free(h_C);
}
