#include <iostream>
#include <fstream>
#include <string.h>
#include <cmath>
#include "Graph.h"
#include "mpi.h"

#define COUT false

using namespace std;

void guardaEnArchivo(int n, double t)
{
    ofstream archivo ("output/floyd2D.dat" , ios_base::app | ios_base::out);
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

    if(idProceso == 0){
        G = new Graph();
        G->lee(argv[1]);
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

    // Obtenemos la raiz de p y el tamaño de un bloque en una dimension
    // Asumimos que el numero de vertices N es multiplo de la raiz del numero de procesos P.
    int sqrtP = sqrt(numeroProcesos),
        tamBloque = nverts / sqrtP;

    // <== COMUNICADORES
    // ==============================>
    /*
        Ejemplo: N = 4 P = 4   -> sqrtP = 2
        [   P0    |   P1    ]   
        [   P2    |   P3    ]
    
        | P0 | P1 | P2 | P3 |
        ---------------------   Para P1
        | 0  | 0  | 1  | 1  |   idHorizontal = 1/2 = 0
        | 0  | 1  | 0  | 1  |   idVertical   = 1%2 = 1

        Equivale a:
            CommHorizontal
            -> [    0     |   0     ]  P0 y P1 estan en la misma fila
            -> [    1     |   1     ]  P2 y P3 estan en la misma fila

            CommVertical                 
            [    0     |   1     ]  P0 y P2 estan en la misma columna
            [    0     |   1     ]  P1 y P3 estan en la misma columna
                 ^         ^
    */
    int idHorizontal = idProceso / sqrtP,
        idVertical = idProceso % sqrtP,
        idProcesoHorizontal,
        idProcesoVertical;
    
    MPI_Comm commHorizontal, commVertical;

    // Creamos los comunicadores, los procesos con el mismo idHorizontal entraran en el mismo comunicador, igual para idVertical
    MPI_Comm_split(MPI_COMM_WORLD, idHorizontal, idProceso, &commHorizontal);
    MPI_Comm_split(MPI_COMM_WORLD, idVertical, idProceso, &commVertical);

    // Obtenemos el nuevo rango asignado dentro de commHorizontal y commVertical
    MPI_Comm_rank(commHorizontal, &idProcesoHorizontal);
    MPI_Comm_rank(commVertical, &idProcesoVertical);

    // <== EMPAQUETAR DATOS
    // ============================================>
    MPI_Datatype MPI_BLOQUE;
    int bufferSalida[nverts*nverts];
    int filaSubmatriz, columnaSubmatriz, comienzo;

    if (idProceso == 0)
    {
        // Definimos bloque como una matriz cuadrada de tamaño tamBloque
        MPI_Type_vector( tamBloque, tamBloque, nverts, MPI_INT, &MPI_BLOQUE );

        // Se hace publico el nuevo tipo de dato
        MPI_Type_commit(&MPI_BLOQUE);

        int posActualBuffer = 0;

        // Empaqueta bloque a bloque en el buffer de envio
        for (int i = 0; i < numeroProcesos; i++)
        {
            filaSubmatriz = i / sqrtP;
            columnaSubmatriz = i % sqrtP;
            comienzo = columnaSubmatriz * tamBloque + filaSubmatriz * tamBloque * tamBloque * sqrtP;

            MPI_Pack( G->getPtrMatriz() + comienzo, 1, MPI_BLOQUE, bufferSalida, sizeof(int) * nverts * nverts, &posActualBuffer, MPI_COMM_WORLD );
        }

        // Liberamos el tipo de dato creado
        MPI_Type_free(&MPI_BLOQUE);
    }
    
    // <== REPARTO DE LA MATRIZ A LOS PROCESOS
    // ============================================>
    int subMatriz[tamBloque][tamBloque];

    // Repartimos los valores del grafo entre los procesos
    MPI_Scatter( bufferSalida, sizeof(int) * tamBloque * tamBloque, MPI_PACKED, subMatriz, tamBloque * tamBloque, MPI_INT, 0, MPI_COMM_WORLD );    

    // <== FLOYD
    // ============================================>  
    int i, j, k, vikj, iGlobal, jGlobal,
        // Principio y fin filas del proceso
        iLocalInicio = idHorizontal * tamBloque, 
        iLocalFinal = (idHorizontal + 1) * tamBloque,
        // Principio y fin columnas del proceso
        jLocalInicio = idVertical * tamBloque,
        jLocalFinal = (idVertical + 1) * tamBloque,
        idProcesoBloqueK = 0,
        indicePartidaFilaK = 0;

    int * filak = new int[tamBloque], * columnak = new int[tamBloque];
    
    for(i = 0; i<tamBloque; i++)
    {
        filak[i] = 0;
        columnak[i] = 0;
    }

    // Iniciamos el cronometro
    double t = MPI_Wtime();

    #if !COUT
        cout.setstate(ios_base::failbit);
    #endif
    for(k = 0; k<nverts; k++)
    {
        // idHorizontal y idVertical del comunicador horizontal y vertical
        idProcesoBloqueK = k / tamBloque;
        indicePartidaFilaK = k % tamBloque;

        if (k >= iLocalInicio && k < iLocalFinal)
        {
            copy(subMatriz[indicePartidaFilaK], subMatriz[indicePartidaFilaK] + tamBloque, filak);
        }
        
        if (k >= jLocalInicio && k < jLocalFinal)
        {
            for (i = 0; i < tamBloque; i++)            
                columnak[i] = subMatriz[i][indicePartidaFilaK];
            
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(filak, tamBloque, MPI_INT, idProcesoBloqueK, commVertical);
        MPI_Bcast(columnak, tamBloque, MPI_INT, idProcesoBloqueK, commHorizontal);

        for(i = 0; i<tamBloque; i++)
        {
            iGlobal = iLocalInicio + i;    
            for(j = 0; j<tamBloque; j++)
            {
                jGlobal = jLocalInicio + j;
                if (iGlobal != jGlobal && iGlobal != k && jGlobal != k)  // No iterar sobre celdas de valor 0
                {   
                    vikj = columnak[i] + filak[j];
                    vikj = min(vikj, subMatriz[i][j]);
                    subMatriz[i][j] = vikj;
                }
            }
        }
    }

    // Paramos el cronometro
    t = MPI_Wtime()-t;

    MPI_Barrier(MPI_COMM_WORLD);

    // <== RECOPILAR LOS RESULTADOS
    // ====================================>
    MPI_Gather( subMatriz, tamBloque * tamBloque, MPI_INT, bufferSalida, sizeof(int) * tamBloque * tamBloque, MPI_PACKED, 0, MPI_COMM_WORLD );

    // <== DESEMPAQUETAR DATOS
    // ============================================>
    if (idProceso == 0)
    {
        MPI_Type_vector(tamBloque, tamBloque, nverts, MPI_INT, &MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);
        
        int posicion = 0;

        for (int i = 0; i < numeroProcesos; i++) {
            filaSubmatriz = i / sqrtP;
            columnaSubmatriz = i % sqrtP;
            comienzo = columnaSubmatriz * tamBloque + filaSubmatriz * tamBloque * tamBloque * sqrtP;

            MPI_Unpack( bufferSalida, sizeof(int) * nverts * nverts, &posicion, G->getPtrMatriz() + comienzo, 1, MPI_BLOQUE,  MPI_COMM_WORLD );
        }

        MPI_Type_free(&MPI_BLOQUE);
    }

    // <== MOSTRAR RESULTADOS
    // ============================================>  
    if(idProceso == 0){

        #if !COUT
            cout.clear();
        #endif
        cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl;
        G->imprime();
        cout << endl << "Tiempo gastado = "<< t << endl << endl;

        guardaEnArchivo(nverts, t);

        // Elimino el puntero de ptrInicioMatriz
        ptrInicioMatriz = NULL;
        delete ptrInicioMatriz;

        // Elimino el objeto grafo usando su destructor
        delete G;
    }

    MPI_Finalize();
    
    delete [] filak;
    delete [] columnak;

    return EXIT_SUCCESS; 
}