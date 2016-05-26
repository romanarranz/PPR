#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string.h>
#include <time.h>
#include <sstream>
using std::cout;
using std::cerr;
using std::endl;
using std::min;

#include "cuda_runtime.h"

#include "Graph.h"
#include "floyd.h"

// Error handling macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void guardarArchivo(std::string outputFile, int n, double t){
    std::ofstream archivo (outputFile.c_str(), std::ios_base::app | std::ios_base::out);
    if (archivo.is_open()){
        std::stringstream ns, ts;
        ns << n;
        ts << t;
        std::string input =  ns.str() + "\t" + ts.str() + "\n";
        archivo << input;
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

void copiaGrafo(int * h_M, Graph g, int N){
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            h_M[i * N + j] = g.arista(i,j);
}

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

    // 16 * 16 = 256 threads en un bloque, cumple que es multiplo de 32 y es con el que mejor resultados he obtenido
    short blockS = 16;

    // CPU variables
    Graph G;
    G.lee(argv[1]);
    // G.imprime();

    const unsigned int N = G.vertices;
    const unsigned int sizeMatrix = N * N;
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
    double Tgpu = clock();
    Tgpu = (Tgpu-t1)/CLOCKS_PER_SEC;

    cout << "CPU: Mostrando resultados..." << endl;
    cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl << endl;
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

    // Guardar en el archivo los resultados
    std::string archivo = "output/floyd2D.dat";
    guardarArchivo(archivo, N, Tgpu);

    // Liberando memoria de CPU
    free(h_M);
}
