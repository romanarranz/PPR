#include <iostream>
#include <stdlib.h>
#include <math.h>
using std::cout;
using std::endl;

#include "sumaVectores.h"

int main(){

    int N = 1024;
    int size = N * sizeof(float);

    // CPU variables
    float * h_A = (float *) malloc(size);
    float * h_B = (float *) malloc(size);
    float * h_C = (float *) malloc(size);

    // CPU function
    initVectors(h_A, h_B, h_C, N);

    // <== Bloques 1D
    // ==============================>
    int numThreadsBloque = 256;
    int numBloques = ceil( (float) N / numThreadsBloque);

    computeGPU(h_A, h_B, h_C, N, numBloques, numThreadsBloque);

    cout << "CPU: Mosrando resultados..." << endl;
    cout << "h_C[0] = " << h_C[0] << " ... h_C[N-1] = " << h_C[N-1] << endl;

    // Liberando memoria de CPU
    free(h_A);
    free(h_B);
    free(h_C);
}
