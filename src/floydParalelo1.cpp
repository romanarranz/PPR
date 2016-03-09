#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include "Graph.h"
#include "mpi.h"

using namespace std;

int main (int argc, char *argv[])
{
    int numeroProcesos, idProceso;

    int nverts, *ptrInicioMatriz = NULL;
    Graph * G;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    if (argc != 2) 
	{
	   cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	   return(-1);
	}

    if(idProceso == 0){
        G = new Graph();
        G->lee(argv[1]);		// Read the Graph
        cout << "EL Grafo de entrada es:"<<endl;
        G->imprime();

        nverts = G->vertices;
        ptrInicioMatriz = G->getPtrMatriz();
    }

    // Todos los procesos deben de conocer el nverts
    MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int tamVectorLocal = (nverts*nverts)/numeroProcesos;
    int * vectorLocal = new int[tamVectorLocal];

    // Repartimos los valores del grafo entre los procesos
    MPI_Scatter(
        ptrInicioMatriz,        // Valores a compartir
        tamVectorLocal,         // Cantidad que se envia a cada proceso
        MPI_INT,                // Tipo del dato que se enviara
        vectorLocal,            // Variable donde recibir los datos
        tamVectorLocal,         // Cantidad que recibe cada proceso
        MPI_INT,                // Tipo del dato que se recibira
        0,                      // Proceso que reparte los datos al resto (En este caso es P0)
        MPI_COMM_WORLD
    );

    // Iniciamos el cronometro
    double t = MPI_Wtime();

    // Todos los procesos conocen i,j,k,vijk
    int i, j, k, vikj;
    
    // Cada proceso tendra su correspondencia local de iteraciones
    int iLocalInicio = idProceso*(nverts/numeroProcesos),
        iLocalFinal = (idProceso+1)*(nverts/numeroProcesos);
    
    // Todos los procesos deben conocer la filak
    vector<int> filak;
    filak.resize(nverts, 0);
    
    for(k = 0; k<nverts; k++)
    {
        
        if(idProceso == 0)
            filak = G->getFilaK(k);
        
        // Compartimos la fila k entre todos los procesos        
        MPI_Bcast(&filak[0], nverts, MPI_INT, 0, MPI_COMM_WORLD);

        for(i = iLocalInicio; i<iLocalFinal; i++)
        {
            for(j = 0; j<nverts; j++)
            {
                if (i!=j && i!=k && j!=k) 
                {   
                    vikj = vectorLocal[(i*nverts)%tamVectorLocal + k] + filak[j];
                    vikj = min(vikj, vectorLocal[(i * nverts)%tamVectorLocal + j]);
                    vectorLocal[(i*nverts)%tamVectorLocal + j] = vikj;
                }
            }
        }

        if(idProceso == 0)
            filak.clear();
    }

    // Recogemos todos los datos en P0
    MPI_Gather(
        &vectorLocal[0],
        tamVectorLocal,
        MPI_INT,
        ptrInicioMatriz,
        tamVectorLocal,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Paramos el cronometro
    t = MPI_Wtime()-t;

    MPI_Barrier(MPI_COMM_WORLD);

    if(idProceso == 0){

        cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl;
        G->imprime();
        cout << endl << "Tiempo gastado= "<< t << endl << endl;

        // Elimino el puntero de ptrInicioMatriz
        ptrInicioMatriz = NULL;
        delete ptrInicioMatriz;

        // Elimino el objeto grafo usando su destructor
        delete G;
    }

    MPI_Finalize();  
}