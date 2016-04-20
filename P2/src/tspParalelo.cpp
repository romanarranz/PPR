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
MPI_Comm COMM_EQUILIBRADO_CARGA, COMM_DETECCION_FIN;
int id, P;

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

void LeerProblemaInicial(tNodo * nodo){
	cout << "LeerProblemaInicial" << endl;
}

void EquilibrarCarga(tPila * pila, bool fin){
	int solicitante, flag, PETICION = 0, NODOS = 1;
	MPI_Status estado;

	if(pila->vacia()){ // el proceso no tiene trabajo: pide a otros procesos

		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		MPI_Send(&id, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);

		while(pila->vacia() && !fin){

			//ESPERAR MENSAJE DE OTRO PROCESO
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &estado);

			switch(estado.MPI_TAG){
				case 0: 	// peticion de trabajo

					//RECIBIR MENSAJE DE PETICION DE TRABAJO
					MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &estado);

					if(solicitante == id){ // peticion devuelta
						//REENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);

						//INICIAR DETECCION DE POSIBLE SITUACION DE FIN ??????????

					}
					else{ // peticion de otro proceso: la retransmite al siguiente
						//PASAR LA PETICION AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);
					}
					break;

				case 1:		// resultado de una peticion de trabajo
					int cantidadNodos;
					// OBTENER LA CANTIDAD DE NODOS QUE SE HAN DONADO
					MPI_Get_count(&estado, MPI_INT, &cantidadNodos);

					//RECIBIR NODOS DEL PROCESO DONANTE EN LA PILA
					MPI_Recv(&pila->nodos, cantidadNodos, MPI_INT, MPI_ANY_SOURCE, NODOS, COMM_EQUILIBRADO_CARGA, &estado);
					pila->tope = cantidadNodos;
					break;
			}
		}

		// El proceso tiene nodos para trabajar
		if(!fin){
			tPila pila2;
			pila->divide(pila2);

			// sondear si hay mensajes pendientes de otros procesos
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &estado);

			while(flag){ // atiende peticiones mientras haya mensajes

				// RECIBIR MENSAJE DE PETICION DE TRABAJO
				MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &estado);

				if(pila->tope > 1){  // hay suficientes nodos en la pila para ceder

					// ENVIAR NODOS AL PROCESO SOLICITANTE
					MPI_Send(&pila2.nodos[0], pila2.tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
				}
				else {
					// PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
					MPI_Send(&solicitante, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);
				}

				// sondear si hay mensajes pendientes de otros procesos
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &estado);
			}
		}
	}
}

int main (int argc, char **argv) {

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

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

	int  U;             	// valor de c.s.
	int iteraciones = 0;
	tPila * pila = new tPila();         	// pila de nodos a explorar

	// solo P0 reserva memoria a la matriz
	if(id == 0){
		tsp0 = reservarMatrizCuadrada(NCIUDADES);
	}

	// <== Inicializaciones de todos los procesos
	// ========================================>
	U = INFINITO;         	// inicializa cota superior
	InicNodo (&nodo);       // inicializa estructura nodo

	// solo P0 rellena la matriz
	if(id == 0){
		LeerMatriz (argv[2], tsp0);    // lee matriz de fichero
		fin = Inconsistente(tsp0);
	}

	MPI_Bcast(&fin, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

	if(id != 0){
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

		// MPI_Bcast(U);
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
