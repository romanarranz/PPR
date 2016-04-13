/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string.h>
#include "mpi.h"
#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;

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

void LeerProblemaInicial(Nodo * nodo){
	cout << "LeerProblemaInicial" << endl;
}

void EquilibrarCarga(tPila * pila, bool fin){
	if(Vacia(pila)){ // el proceso no tiene trabajo: pide a otros procesos
		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		while(Vacia(pila) && !fin){
			//ESPERAR MENSAJE DE OTRO PROCESO
			switch(tipo de mensaje){
				case PETIC: 	// peticion de trabajo
					//RECIBIR MENSAJE DE PETICION DE TRABAJO
					if(solicitante == id){ // peticion devuelta
						//REENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
						//INICIAR DETECCION DE POSIBLE SITUACION DE FIN
					}
					else // peticio de otro proces: la retransmite al siguiente
						//PASAR LA PETICION AL PROCESO (id+1)%P
					break;

				case NODOS:		// resultado de una peticion de trabajo
					//RECIBIR NODOS DEL PROCESO DONANTE
					//ALMACENAR NODOS RECIBIDOS EN LA PILA
					break;
			}
		}

		// El proceso tiene nodos para trabajar
		if(!fin){
			// sondear si hay mensajes pendientes de otros procesos
			while(hay mensajes){ // atiende peticiones mientras haya mensajes
				// recibir mensaje de peticion de trabajo
				if(hay suficientes nodos en la pila para ceder)
					// enviar nodos al proceso solicitante
				else
					// pasar peticion de trabajo al proceso (id+1)%P

				// sondear si hay mensajes pendientes de otros procesos
			}
		}
	}
}

void Pop(tPila * pila, Nodo nodo){
	pila.pop(nodo);
}

int main (int argc, char **argv) {

	int numeroProcesos, idProceso;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesos);
	MPI_Comm_rank(MPI_COMM_WORLD, &idProceso);

	switch (argc) {
		case 3:
			NCIUDADES = atoi(argv[1]);
			break;

		default:
			cerr << "La sintaxis es: bbseq <tamano> <archivo>" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
			break;
	}

	// <== Valores que conocen todos los procesos
	// ========================================>
	int ** tsp0 = 0;
	tNodo 	nodo,         	// nodo a explorar
			lnodo,        	// hijo izquierdo
			rnodo,        	// hijo derecho
			solucion;     	// mejor solucion
	bool 	activo,        	// condicion de fin
			nueva_U;       	// hay nuevo valor de c.s.

	int  U;             	// valor de c.s.
	int iteraciones = 0;
	tPila pila;         	// pila de nodos a explorar

	// solo P0 reserva memoria a la matriz
	if(idProceso == 0){
		tsp0 = reservarMatrizCuadrada(NCIUDADES);
	}

	// <== Inicializaciones de todos los procesos
	// ========================================>
	U = INFINITO;         	// inicializa cota superior
	InicNodo (&nodo);       // inicializa estructura nodo

	// solo P0 rellena la matriz
	if(idProceso == 0){
		LeerMatriz (argv[2], tsp0);    // lee matriz de fichero
		activo = !Inconsistente(tsp0);
	}

	MPI_Bcast(&activo, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

	if(idProceso == 0)
		LeerProblemaInicial()

    double t = MPI::Wtime();

	// ciclo del Branch&Bound
	while (activo) {

		Ramifica (&nodo, &lnodo, &rnodo, tsp0);
		nueva_U = false;

		if (Solucion(&rnodo)) {
			// se ha encontrado una solucion mejor
			if (rnodo.ci() < U) {
				U = rnodo.ci();
				nueva_U = true;
				CopiaNodo (&rnodo, &solucion);
			}
		}
		//  no es un nodo solucion
		else {
			//  cota inferior menor que cota superior
			if (rnodo.ci() < U) {
				if (!pila.push(rnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		if (Solucion(&lnodo)) {
			// se ha encontrado una solucion mejor
			if (lnodo.ci() < U) {
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo (&lnodo,&solucion);
			}
		}
		// no es nodo solucion
		else {
			// cota inferior menor que cota superior
			if (lnodo.ci() < U) {
				if (!pila.push(lnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		// MPI_Bcast(U);
		if (nueva_U) pila.acotar(U);

		EquilibrarCarga(&pila, &activo);
		if(activo) Pop(&pila, &nodo);

		iteraciones++;
	}

    t = MPI::Wtime()-t;

	// <== Mostramos resultados
	// ========================================>
	if(idProceso == 0){
		printf ("Solucion: \n");

		EscribeNodo(&solucion);
		cout<< "Tiempo gastado= "<<t<<endl;
		cout << "Numero de iteraciones = " << iteraciones << endl << endl;

		// Liberamos memoria
		liberarMatriz(tsp0);
	}

    MPI::Finalize();
}
