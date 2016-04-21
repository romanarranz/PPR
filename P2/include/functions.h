#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include "mpi.h"
#include "libbb.h"

using namespace std;

extern int id, P;
extern MPI_Comm COMM_EQUILIBRADO_CARGA, COMM_DETECCION_FIN;
extern int U; // cota superior de cada proceso
extern bool difundir_cs_local, pendiente_retorno_cs;
extern int flag, solicitante;
extern MPI_Status status;
extern const int PETICION, NODOS;

void EquilibrarCarga(tPila * pila, bool *fin);
void DifusionCotaSuperior();

#endif
