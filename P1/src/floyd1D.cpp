#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

#define COUT false

using namespace std;

void guardaEnArchivo(int n, double t)
{
    ofstream archivo ("output/floyd1D.dat" , ios_base::app | ios_base::out);
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
    int numeroProcesos, idProceso;

    int nverts, *ptrInicioMatriz = NULL;
    Graph * G;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

    if (argc != 2) 
	{
	   cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	   return EXIT_FAILURE;
	}

    #if !COUT
        cout.setstate(ios_base::failbit);
    #endif
    if(idProceso == 0)
    {
        G = new Graph();
        G->lee(argv[1]);		// Read the Graph        
        cout << "EL Grafo de entrada es:"<<endl;
        G->imprime();

        nverts = G->vertices;
        ptrInicioMatriz = G->getPtrMatriz();        
    }

    // Todos los procesos deben de conocer el nverts
    MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(nverts%numeroProcesos != 0)
    {
        cout << "P"<< idProceso<<" -> El numero de vertices no es divisible entre el numero de procesos" << endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // Todos los procesos conocen su vector local
    int tamVectorLocal = (nverts*nverts)/numeroProcesos,
        tamFilaLocal = tamVectorLocal/nverts;
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

    // Todos los procesos conocen i,j,k,vijk
    int i, j, k, vikj;
    
    // Cada proceso tendra su correspondencia local de iteraciones
    int iLocalInicio = idProceso*(nverts/numeroProcesos),
        iLocalFinal = (idProceso+1)*(nverts/numeroProcesos),
        idProcesoBloqueK = 0,
        indicePartidaFilaK = 0;

    // Todos los procesos conocen su filak
    int * filak = new int[nverts];
    for(i = 0; i<nverts; i++)
        filak[i] = 0;

    int aux, locali, vij;

    // Iniciamos el cronometro
    double t = MPI_Wtime();
    
    cout << "P" << idProceso << endl;
    for(k = 0; k<nverts; k++)
    {
        // Cada proceso debe mandar su filak a todos
        /*
        Ejemplo: P = 3 y N = 6
                    --------------------------
            P = 0 k:0   -> indice 0 del vectorLocal de P0
                  k:1   -> indice 6 del vectorLocal de P0
                    --------------------------  
            P = 1 k:2   -> indice 0 del vectorLocal de P1   
                  k:3   -> indice 6 del vectorLocal de P1
                    --------------------------
            P = 2 k:4   -> indice 0 del vectorLocal de P2
                  k:5   -> indice 6 del vectorLocal de P2
                    --------------------------
        */
        idProcesoBloqueK = k / tamFilaLocal;
        //indicePartidaFilaK = (k*nverts)%tamVectorLocal;
        indicePartidaFilaK = k - iLocalInicio;

        if(k >= iLocalInicio && k < iLocalFinal)
        {
            cout << "\tk: " << k << ", iLocalInicio: " << iLocalInicio << ", iLocalFinal: "<< iLocalFinal << endl;            
            cout << "filak: ";
            for(i = 0; i<nverts; i++)
            {
                filak[i] = vectorLocal[indicePartidaFilaK + i];
                cout << filak[i] << ",";
            }
            cout << endl;
        }
        
        MPI_Bcast(&filak[0], nverts, MPI_INT, idProcesoBloqueK, MPI_COMM_WORLD);
        
        for(i = iLocalInicio; i<iLocalFinal; i++)
        {
            locali = i - iLocalInicio;
            //aux = (i*nverts)%tamVectorLocal;
            aux = locali * nverts;
            for(j = 0; j<nverts; j++)
            {
                vij = aux + j;
                // no iterar sobre la diagonal de la matriz
                if (i!=j && i!=k && j!=k) 
                {   
                    vikj = vectorLocal[ aux + k] + filak[j];
                    vikj = min(vikj, vectorLocal[vij]);
                    vectorLocal[vij] = vikj;
                }
            }
        }
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

    if(idProceso == 0)
    {
        
        cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl;
        G->imprime();
        #if !COUT
            cout.clear();
        #endif
        cout << endl << "Tiempo gastado = "<< t << endl << endl;

        guardaEnArchivo(nverts, t);

        // Elimino el puntero de ptrInicioMatriz
        ptrInicioMatriz = NULL;
        delete ptrInicioMatriz;

        // Elimino el objeto grafo usando su destructor
        delete G;
    }

    // Cada proceso elimina su filak y su vector local
    delete [] filak;
    delete [] vectorLocal;

    MPI_Finalize();  

    return EXIT_SUCCESS;
}