#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

#define COUT false

using namespace std;

void guardaEnArchivo(int n, double t)
{
    ofstream archivo ("output/floydS.dat" , ios_base::app | ios_base::out);
    if (archivo.is_open())
    {
        archivo << to_string(n) + "\t" + to_string(t) + "\n";
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

int main (int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    if (argc != 2)
    {
	   cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	   return EXIT_FAILURE;
	}

    Graph * G = new Graph();
    G->lee(argv[1]);		// Read the Graph

    #if !COUT
        cout.setstate(ios_base::failbit);
    #endif
    cout << "EL Grafo de entrada es:"<<endl;
    G->imprime();

    int nverts = G->vertices;

    double t = MPI_Wtime();

    // BUCLE PPAL DEL ALGORITMO
    int i,j,k,vikj;
    for(k = 0; k<nverts; k++)
    {
        for(i = 0; i<nverts; i++)
        {
            for(j = 0; j<nverts; j++)
            {
                if (i != j && i != k && j != k) 
                {
                    vikj = G->arista(i,k)+G->arista(k,j);
                    vikj = min(vikj,G->arista(i,j));
                    G->inserta_arista(i,j,vikj);
                }
            }
        }
    }
    
    t = MPI_Wtime() - t;
    MPI_Finalize();

    cout << endl << "EL Grafo con las distancias de los caminos más cortos es:" << endl << endl;
    G->imprime();
    #if !COUT
        cout.clear();
    #endif
    cout << endl << "Tiempo gastado = "<< t << endl << endl;
    guardaEnArchivo(nverts, t);

    delete G;

    return EXIT_SUCCESS;
}