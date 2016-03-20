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
        idProcesoVertical,
        numProcesosHorizontal,
        numProcesosVertical;
    
    MPI_Comm commHorizontal, commVertical;

    // Creamos los comunicadores, los procesos con el mismo idHorizontal entraran en el mismo comunicador, igual para idVertical
    MPI_Comm_split(MPI_COMM_WORLD, idHorizontal, idProceso, &commHorizontal);
    MPI_Comm_split(MPI_COMM_WORLD, idVertical, idProceso, &commVertical);

    // Obtenemos el nuevo rango asignado dentro de commHorizontal y commVertical
    MPI_Comm_rank(commHorizontal, &idProcesoHorizontal);
    MPI_Comm_size(commHorizontal, &numProcesosHorizontal);

    MPI_Comm_rank(commVertical, &idProcesoVertical);
    MPI_Comm_size(commVertical, &numProcesosVertical);

    // <== EMPAQUETAR DATOS
    // ============================================>
    MPI_Datatype bloque;
    int * bufferSalida = new int[nverts * nverts];
    int filaSubmatriz, columnaSubmatriz, posInicio;

    if (idProceso == 0)
    {
        // Definimos bloque como una matriz cuadrada de tamaño tamBloque
        MPI_Type_vector(
            tamBloque,  // tamaño del bloque
            tamBloque,  // separador entre bloque y bloque
            nverts,     // cantidad de bloques que cogemos (filas)
            MPI_INT,    // tipo de dato origen
            &bloque     // tipo de dato destino
        );

        // Se hace publico el nuevo tipo de dato
        MPI_Type_commit(&bloque);

        int posActualBuffer = 0;
        for (int i = 0; i < numeroProcesos; i++)
        {
            filaSubmatriz = i / sqrtP;
            columnaSubmatriz = i % sqrtP;
            posInicio = columnaSubmatriz * tamBloque + filaSubmatriz * tamBloque * tamBloque * sqrtP;

            MPI_Pack(
                G->getPtrMatriz() + posInicio,       // posicion de partida
                1,                                  // numero de datos de entrada
                bloque,                             // tipo de dato de los datos de entrada
                bufferSalida,                       // buffer de salida
                sizeof(int) * nverts * nverts,      // tamaño del buffer de salida en bytes
                &posActualBuffer,                   // posicion actual del buffer de salida en bytes
                MPI_COMM_WORLD
            );
        }

        // Liberamos el tipo de dato creado
        MPI_Type_free(&bloque);
    }
    
    // <== REPARTO DE LA MATRIZ A LOS PROCESOS
    // ============================================>
    int * subMatriz = new int[tamBloque * tamBloque],
          tamSubmatriz = tamBloque * tamBloque;

    // Repartimos los valores del grafo entre los procesos
    MPI_Scatter(
        &bufferSalida[0],                       // Valores a compartir
        sizeof(int) * tamBloque * tamBloque,    // Cantidad que se envia a cada proceso
        MPI_PACKED,                             // Tipo del dato que se enviara
        subMatriz,                              // Variable donde recibir los datos
        sizeof(int) * tamBloque * tamBloque,    // Cantidad que recibe cada proceso
        MPI_INT,                                // Tipo del dato que se recibira
        0,                                      // Proceso que reparte los datos al resto (En este caso es P0)
        MPI_COMM_WORLD
    );

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

    int * filak = new int[tamBloque],
        * columnak = new int[tamBloque];
    
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
        idProcesoBloqueK = k / tamSubmatriz;
        indicePartidaFilaK = (k*tamBloque) % tamSubmatriz;


        if (k >= iLocalInicio && k < iLocalFinal)
        {
            cout << "\t filak:" << k << " -> ";
            for(i = 0; i<tamBloque; i++)
            {
                filak[i] = subMatriz[indicePartidaFilaK + i];
                cout << filak[i] << ",";
            }
            cout << endl;
        }
        
        if (k >= jLocalInicio && k < jLocalFinal)
        {
            cout << "\t columnak:" << k << " -> ";
            for (i = 0; i < tamBloque*tamBloque; i+=tamBloque)
            {
                columnak[i] = subMatriz[k%tamBloque + i];
                cout << columnak[i] << ",";
            }
            cout << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&filak[0], tamBloque, MPI_INT, idProcesoBloqueK, commVertical);
        MPI_Bcast(&columnak[0], tamBloque, MPI_INT, idProcesoBloqueK, commHorizontal);

        for(i = 0; i<tamBloque; i++)
        {
            iGlobal = iLocalInicio + i;    
            for(j = 0; j<tamBloque; j++)
            {
                jGlobal = jLocalInicio + j;
                if (iGlobal != jGlobal && iGlobal != k && jGlobal != k)  // No iterar sobre celdas de valor 0
                {   
                    vikj = columnak[i] + filak[j];
                    vikj = min(vikj, subMatriz[(i*tamBloque)%tamSubmatriz + j]);
                    subMatriz[(i*tamBloque)%tamSubmatriz + j] = vikj;
                }
            }
        }
    }

    // Paramos el cronometro
    t = MPI_Wtime()-t;

    MPI_Barrier(MPI_COMM_WORLD);

    // <== RECOPILAR LOS RESULTADOS
    // ====================================>
    MPI_Gather(
        &subMatriz[0],
        tamBloque * tamBloque,
        MPI_INT,
        &bufferSalida[0],
        nverts * nverts,
        MPI_PACKED,
        0,
        MPI_COMM_WORLD
    );

    // <== DESEMPAQUETAR DATOS
    // ============================================>
    if (idProceso == 0)
    {
        MPI_Type_vector(tamBloque, tamBloque, nverts, MPI_INT, &bloque);
        MPI_Type_commit(&bloque);

        for (int i = 0, posicion = 0; i < numeroProcesos; i++) {
            filaSubmatriz = i / sqrtP;
            columnaSubmatriz = i % sqrtP;
            posInicio = columnaSubmatriz * tamBloque + filaSubmatriz * tamBloque * tamBloque * sqrtP;

            MPI_Unpack(
                bufferSalida,
                sizeof(int) * nverts * nverts, 
                &posicion,
                G->getPtrMatriz() + posInicio,
                1,
                bloque, 
                MPI_COMM_WORLD
            );
        }
        MPI_Type_free(&bloque); // Se libera el tipo bloque
    }
    
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

    // Cada proceso elimina su datos locales
    delete [] filak;
    delete [] columnak;
    delete [] subMatriz;
    delete [] bufferSalida;

    MPI_Finalize(); 

    return EXIT_SUCCESS; 
}