#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string.h>
#include "mpi.h"
#include "libbb.h"
#include "functions.h"

using namespace std;

unsigned int NCIUDADES;
MPI_Comm COMM_EQUILIBRADO_CARGA, COMM_DETECCION_FIN;
int id, P;
int U;						// valor de cota superior
int solicitante, flag;
MPI_Status status;

void guardaEnArchivo(int n, double t)
{
    ofstream archivo ("output/tspP.dat" , ios_base::app | ios_base::out);
    if (archivo.is_open())
    {
        archivo << to_string(n) + "\t" + to_string(t) + "\n";
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

int main (int argc, char **argv) {

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(argc < 2 && argc > 4){
		cerr << "La sintaxis es: mpirun -np P bin/tspParalelo <tamano> <archivo>" << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}
	else {
		NCIUDADES = atoi(argv[1]);
	}

	// Duplicamos el comunicador para el proceso de deteccion de fin
	MPI_Comm_dup(MPI_COMM_WORLD, &COMM_EQUILIBRADO_CARGA);
	MPI_Comm_dup(MPI_COMM_WORLD, &COMM_DETECCION_FIN);

	// <== Valores que conocen todos los procesos
	// ========================================>
	int ** tsp0 = 0;
	tNodo 	nodo,         	// nodo a explorar
			nodoIzq,        // hijo izquierdo
			nodoDer,        // hijo derecho
			solucion;     	// mejor solucion
	bool 	fin,        	// condicion de fin
			nueva_U;       	// hay nuevo valor de c.s.

	int iteraciones = 0;
	tPila * pila = new tPila();         	// pila de nodos a explorar

	// <== Inicializaciones de todos los procesos
	// ========================================>
	U = INFINITO;         	// inicializa cota superior
	InicNodo (&nodo);       // inicializa estructura nodo

	// solo P0 rellena la matriz
	if(id == 0){
		tsp0 = reservarMatrizCuadrada(NCIUDADES); 	// reserva memoria a la matriz
		LeerMatriz (argv[2], tsp0);    				// lee matriz de fichero
		fin = Inconsistente(tsp0);

		MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&fin, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	}
	else {
		EquilibrarCarga(pila, &fin);
		if(!fin) pila->pop(nodo);
	}

    double t = MPI::Wtime();

	// ciclo del Branch&Bound
	while (!fin) {

		Ramifica (&nodo, &nodoIzq, &nodoDer, tsp0);
		nueva_U = false;

		if (Solucion(&nodoDer)) {
			// se ha encontrado una solucion mejor
			if (nodoDer.ci() < U) {
				U = nodoDer.ci();
				nueva_U = true;
				CopiaNodo (&nodoDer, &solucion);
			}
		}
		//  no es un nodo solucion
		else {
			//  cota inferior menor que cota superior
			if (nodoDer.ci() < U) {
				if (!pila->push(nodoDer)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		if (Solucion(&nodoIzq)) {
			// se ha encontrado una solucion mejor
			if (nodoIzq.ci() < U) {
				U = nodoIzq.ci();
				nueva_U = true;
				CopiaNodo (&nodoIzq,&solucion);
			}
		}
		// no es nodo solucion
		else {
			// cota inferior menor que cota superior
			if (nodoIzq.ci() < U) {
				if (!pila->push(nodoIzq)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		MPI_Bcast(&U, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (nueva_U) pila->acotar(U);

		EquilibrarCarga(pila, &fin);
		if(!fin) pila->pop(nodo);

		iteraciones++;
	}

    t = MPI::Wtime()-t;

	// <== Mostramos resultados
	// ========================================>
	if(id == 0){
		printf ("Solucion: \n");

		EscribeNodo(&solucion);
		cout<< "Tiempo gastado= "<<t<<endl;
		cout << "Numero de iteraciones = " << iteraciones << endl << endl;

		// Liberamos memoria
		liberarMatriz(tsp0);
		delete pila;
	}

    MPI::Finalize();
}
