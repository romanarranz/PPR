#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include "mpi.h"
#include "libbb.h"

using namespace std;

void EquilibrarCarga(tPila * pila, bool *fin);
void DifusionCotaSuperior();

#endif
