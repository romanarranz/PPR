#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string.h>
#include <time.h>
using std::cout;
using std::cerr;
using std::endl;
<<<<<<< HEAD
using std::min;
=======
>>>>>>> master

#include "cuda_runtime.h"

#include "Graph.h"
#include "floyd.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; }

<<<<<<< HEAD
void copiaGrafo(int * h_M, Graph g, int N){
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            h_M[i * N + j] = g.arista(i,j);
}

=======
>>>>>>> master
int main(int argc, char **argv){

    if (argc != 2) {
        cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	    return(-1);
	}

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

<<<<<<< HEAD
    // 16 * 16 = 256 threads en un bloque, cumple que es multiplo de 32 y es con el que mejor resultados he obtenido
    short blockS = 16;
=======
    short blockS = 3;
>>>>>>> master

    // CPU variables
    Graph G;
    G.lee(argv[1]);
    // G.imprime();

    const unsigned int N = G.vertices;
    const unsigned int sizeMatrix = N * N;
<<<<<<< HEAD
    const unsigned int memSize = sizeMatrix * sizeof(int);
    int * h_M = (int *) malloc(memSize);
    copiaGrafo(h_M, G, N);

    dim3 blockSize(blockS, blockS);     // el bloque de 16 * 16 en 2D
    int numThreadsBloque = blockSize.x * blockSize.y;
    dim3 numBlocks ( ceil((float) N/blockSize.x), ceil((float) N/blockSize.y));

    cout << "El blockSize es de: " << blockS << endl;
    cout << "El numBloques es de: " << numBlocks.x * numBlocks.y << endl;
    cout << "El numThreadsBloque es de: " << numThreadsBloque << endl << endl;

    // Calc
    double t1 = clock();
    floyd2DGPU(h_M, N, numBlocks, blockSize);
=======
    int * h_M = G.getMatrix();

    dim3 blockSize(blockS, blockS);
    int numBloques = sizeMatrix / (blockSize.x * blockSize.y);
    int numThreadsBloque = blockSize.x;

    cout << "El blockSize es de: " << blockS << endl;
    cout << "El numBloques es de: " << numBloques << endl;
    cout << "El numThreadsBloque es de: " << numThreadsBloque * numThreadsBloque << endl << endl;

    // Calc
    double t1 = clock();
    floyd2DGPU(h_M, N, numBloques, numThreadsBloque);
>>>>>>> master
    double Tgpu = clock();
    Tgpu = (Tgpu-t1)/CLOCKS_PER_SEC;

    cout << "CPU: Mostrando resultados..." << endl;
    cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl << endl;
<<<<<<< HEAD
    // G.imprime();
    cout << "Tiempo gastado GPU = " << Tgpu << endl << endl;

    // Comprobar si los resultados de CPU y GPU coinciden
    cout << "Comprobando resultados..." << endl;
    int aux, vikj;
    for(int k = 0; k < N; k++)
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                if (i!=j && i!=k && j!=k){
                    aux = G.arista(i,k) + G.arista(k,j);
                    vikj = min(aux, G.arista(i,j));
                    G.inserta_arista(i, j, vikj);
                }

    bool error = false;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if ( abs(h_M[i * N + j] - G.arista(i,j)) > 0 ){
                // cout <<"Error ("<<i<<","<<j<<")   " << h_M[i * N + j] << "..." << G.arista(i,j) << endl;
                error = true;
            }
        }
    }
    if(error) cout <<"With ERRORS" << endl;
    else cout << "ALL OK" << endl;
=======
    G.imprime();
    cout << "Tiempo gastado GPU = " << Tgpu << endl << endl;

    // Comprobar si los resultados de CPU y GPU coinciden
    /*for(int k=0;k<niters;k++)
        for(int i=0;i<nverts;i++)
            for(int j=0;j<nverts;j++)
                if (i!=j && i!=k && j!=k){
                    int vikj=min(G.arista(i,k)+G.arista(k,j),G.arista(i,j));
                    G.inserta_arista(i,j,vikj);
                }

    for(int i=0;i<nverts;i++)
        for(int j=0;j<nverts;j++)
            if (abs(c_Out_M[i*nverts+j]-G.arista(i,j))>0)
                cout <<"Error ("<<i<<","<<j<<")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;*/
>>>>>>> master

    // Liberando memoria de CPU
    free(h_M);
}