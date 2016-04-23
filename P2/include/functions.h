#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define DEBUG_EQUILIBRADO false
#define DEBUG_COTA true
#define DEBUG_DFIN false

#include <iostream>
#include <unistd.h>
#include "mpi.h"
#include "libbb.h"

using namespace std;

// Variables de las tareas
extern int id, P;
extern int siguiente, anterior;
extern int estado;
extern int color;
extern int color_token;
extern int token;
extern bool token_presente;
extern const int ACTIVO, PASIVO;
extern const int BLANCO, NEGRO;

// Comunicadores de procesos
extern MPI_Comm COMM_EQUILIBRADO_CARGA;
extern MPI_Comm COMM_DIFUSION_COTA;

// Variables Difusion Cota Sup
extern int U; // cota superior de cada proceso
extern bool difundir_cs_local, pendiente_retorno_cs;
extern int hay_mensajes, solicitante;

// Variables Mensajes
extern MPI_Status status;
extern const int PETICION, NODOS, TOKEN, FIN;

void EquilibrarCarga(tPila * pila, bool *fin);
void DifusionCotaSuperior(int *U);

#endif
