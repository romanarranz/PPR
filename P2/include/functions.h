#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define DEBUG_EQUILIBRADO true
#define DEBUG_COTA true

#include <iostream>
#include <unistd.h>
#include "mpi.h"
#include "libbb.h"

using namespace std;

extern int id, P;
extern int siguiente, anterior;
extern MPI_Comm COMM_EQUILIBRADO_CARGA;
extern MPI_Comm COMM_DIFUSION_COTA;

extern int U; // cota superior de cada proceso
extern bool difundir_cs_local, pendiente_retorno_cs;
extern int hay_mensajes, solicitante;
extern MPI_Status status;
extern const int PETICION, NODOS;

void EquilibrarCarga(tPila * pila, bool *fin);
void DifusionCotaSuperior(int *U);

#endif
