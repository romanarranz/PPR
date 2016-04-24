/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;

int main (int argc, char **argv) {

MPI::Init(argc,argv);

	switch (argc) {
		case 3:
			NCIUDADES = atoi(argv[1]);
			break;

		default:
			cerr << "La sintaxis es: bbseq <tamano> <archivo>" << endl;
			exit(1);
			break;
	}

	int ** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo 	nodo,         	// nodo a explorar
			lnodo,        	// hijo izquierdo
			rnodo,        	// hijo derecho
			solucion;     	// mejor solucion

	bool 	activo,        	// condicion de fin
			nueva_U;       	// hay nuevo valor de c.s.

	int  U;             	// valor de c.s.
	int iteraciones = 0;

	tPila pila;         	// pila de nodos a explorar

	U = INFINITO;         	// inicializa cota superior
	InicNodo (&nodo);       // inicializa estructura nodo

	LeerMatriz (argv[2], tsp0);    // lee matriz de fichero
	activo = !Inconsistente(tsp0);

    double t=MPI::Wtime();

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

		if (nueva_U) pila.acotar(U);

		activo = pila.pop(nodo);
		iteraciones++;
	}

    t = MPI::Wtime()-t;

    MPI::Finalize();

	printf ("Solucion: \n");

	EscribeNodo(&solucion);
    cout<< "Tiempo gastado= "<<t<<endl;
	cout << "Numero de iteraciones = " << iteraciones << endl << endl;

	liberarMatriz(tsp0);
}
