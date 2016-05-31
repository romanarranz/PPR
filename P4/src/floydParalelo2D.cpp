#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <time.h>
#include "Graph.h"

#include "floyd.h"

using namespace std;

void guardarArchivo(string outputFile, int n, double t){
    ofstream archivo (outputFile.c_str(), ios_base::app | std::ios_base::out);
    if (archivo.is_open()){
        stringstream ns, ts;
        ns << n;
        ts << t;
        string input =  ns.str() + "\t" + ts.str() + "\n";
        archivo << input;
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

void copiaGrafo(int * M, Graph g, int N){
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            M[(i*N)+j] = g.arista(i,j);
}

int main(int argc, char **argv){

    if (argc != 2){
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}

	Graph G;
	G.lee(argv[1]);		// Read the Graph
    // G.imprime();

    const int N = G.vertices;
    const int P = 4;
    if(N % P != 0){
        cerr << "El tamaño del problema no es divisible entre el numero de hebras" << endl;
        return(-1);
    }

    const unsigned int sizeMatrix = N * N;
    const unsigned int memSize = sizeMatrix * sizeof(int);
    int * M = (int *) malloc(memSize);
    copiaGrafo(M, G, N);

    cout << "El numero de vertices es: " << N << endl;
    cout << "Hay " << sizeMatrix << " elementos y se han reservado " << memSize << "B" << endl;

    // Calc
    double tFloyd = floyd2DOpenMP(M, N, P);

    cout << "Mostrando resultados..." << endl;
    cout << "Tiempo gastado = " << tFloyd << endl << endl;

    // Comprobar si los resultados de CPU y GPU coinciden
    cout << "Comprobando resultados..." << endl;
    for(int k=0;k<N;k++){
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                if (i!=j && i!=k && j!=k) {
                    int vikj = min(G.arista(i,k)+G.arista(k,j),G.arista(i,j));
                    G.inserta_arista(i,j,vikj);
                }
            }
        }
    }

    bool error = false;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if ( abs(M[i * N + j] - G.arista(i,j)) > 0 ){
                cout <<"Error ("<<i<<","<<j<<")   " << M[i * N + j] << "..." << G.arista(i,j) << endl;
                error = true;
            }
        }
    }
    if(error) cout << "With ERRORS" << endl;
    else cout << "ALL OK" << endl;

    // Guardar en el archivo los resultados
    std::string archivo = "output/floyd2D.dat";
    guardarArchivo(archivo, N, tFloyd);

    // Liberando memoria de CPU
    free(M);
}
