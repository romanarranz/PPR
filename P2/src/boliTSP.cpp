void Equilibrado_Carga(tPila *pila, bool *fin, tNodo *solucion) {
  color = BLANCO;
  if (pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
    /* Enviar petición de trabajo al proceso (rank + 1) % size */
    MPI_Send(&rank, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);
    while (pila->vacia() && !*fin) {
      /* Esperar mensaje de otro proceso */
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &status);
      switch (status.MPI_TAG) {
        case PETICION: // peticion de trabajo
          /* Recibir mensaje de petición de trabajo */
          MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, comunicadorCarga, &status);
          /* Reenviar petición de trabajo al proceso (rank + 1) % size */
          MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);
          if (solicitante == rank) { // peticion devuelta
            /* Iniciar detección de posible situación de fin */
            estado = PASIVO;
            if (token_presente) {
              if (rank == 0) {
                color_token = BLANCO;
              } else {
                color_token = color;
              }
              /* Enviar Mensaje_testigo a anterior */
              MPI_Send(NULL, 0, MPI_INT, anterior, TOKEN, comunicadorCarga);
              token_presente = false;
              color = BLANCO;
            }
          }
          break;
        case NODOS: // resultado de una petición de trabajo
          MPI_Get_count(&status, MPI_INT, &tamanio);
          /* Recibir nodos del proceso donante */
          MPI_Recv(&pila->nodos[0], tamanio, MPI_INT, MPI_ANY_SOURCE, NODOS, comunicadorCarga, &status);
          /* Almacenar nodos recibidos en la pila */
          pila->tope = tamanio;
          estado = ACTIVO;
          break;
        case TOKEN:
          /* Recibir Mensajes de Petición pendientes */
          MPI_Recv(NULL, 0, MPI_INT, siguiente, TOKEN, comunicadorCarga, &status);
          token_presente = true;
          if (estado == PASIVO) {
            if (rank == 0 && color == BLANCO && color_token == BLANCO) {
              *fin = true;
              /* Enviar Mensaje_fin al proc. siguiente */
              MPI_Send(&solucion->datos[0], 2 * NCIUDADES, MPI_INT, siguiente, FIN, comunicadorCarga);
              /* Recibir Mensaje_fin del proc. anterior */
              posibleSol = new tNodo();
              MPI_Recv(&posibleSol->datos[0], 2 * NCIUDADES, MPI_INT, anterior, FIN, comunicadorCarga, &status);
              if (posibleSol->ci() < solucion->ci()) {
                CopiaNodo(posibleSol, solucion);
              }
              delete posibleSol;
            } else {
              if (rank == 0) {
                color_token = BLANCO;
              } else {
                color_token = color;
              }
              /* Enviar Mensaje_testigo a anterior */
              MPI_Send(NULL, 0, MPI_INT, anterior, TOKEN, comunicadorCarga);
              token_presente = false;
              color = BLANCO;
            }
          }
          break;
        case FIN:
          /* Recibir mensaje de fin */
          *fin = true;
          posibleSol = new tNodo();
          MPI_Recv(&posibleSol->datos[0], 2 * NCIUDADES, MPI_INT, anterior, FIN, comunicadorCarga, &status);
          if (posibleSol->ci() < solucion->ci()) {
            CopiaNodo(posibleSol, solucion);
          }
          delete posibleSol;
          /* Enviar Mensaje_fin al proc. siguiente */
          MPI_Send(&solucion->datos[0], 2 * NCIUDADES, MPI_INT, siguiente, FIN, comunicadorCarga);
          break;
      }
    }
  }
  if (!*fin) { // el proceso tiene nodos para trabajar
    /* Sondear si hay mensajes pendientes de otros procesos */
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &hay_mensajes, &status);
    while (hay_mensajes) { // atiende peticiones mientras haya mensajes
      switch (status.MPI_TAG) {
        case PETICION:
          /* Recibir mensaje de petición de trabajo */
          MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, comunicadorCarga, &status);
          if (pila->tamanio() > 1) {
            /* Enviar nodos al proceso solicitante */
            pilaNueva = new tPila();
            pila->divide(*pilaNueva);
            MPI_Send(&pilaNueva->nodos[0], pilaNueva->tope, MPI_INT, solicitante, NODOS, comunicadorCarga);
            delete pilaNueva;
            if (rank < solicitante) {
              color = NEGRO;
            }
          } else {
            /* Pasar petición de trabajo al proceso (rank + 1) % size */
            MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);
          }
          break;
        case TOKEN:
          /* Recibir Mensaje_testigo de siguiente */
          MPI_Recv(NULL, 0, MPI_INT, siguiente, TOKEN, comunicadorCarga, &status);
          token_presente = true;
          break;
      }
      /* Sondear si hay mensajes pendientes de otros procesos */
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &hay_mensajes, &status);
    }
  }
}

/* ********************************************************************* */

void Difusion_Cota_Superior(int *U, bool *nueva_U) {
  if (difundir_cs_local && !pendiente_retorno_cs) {
    /* Enviar valor local de cs al proceos (rank + 1) % size */
    MPI_Send(&U, 1, MPI_INT, siguiente, 0, comunicadorCota);
    pendiente_retorno_cs = true;
    difundir_cs_local = false;
  }
  /* Sondear si hay mensajes de cota superior pendientes */
  MPI_Iprobe(anterior, MPI_ANY_TAG, comunicadorCota, &hay_mensajes, &status);
  while (hay_mensajes) {
    /* Recibir mensajes con valor de cota superior desde el proceso (rank - 1 + size) % size */
    MPI_Recv(&cs, 1, MPI_INT, anterior, 0, comunicadorCota, &status);
    /* Actualizar valor local de cota superior */
    if (cs < *U) {
      *U = cs;
      *nueva_U = true;
    }
    if (status.MPI_SOURCE == rank && difundir_cs_local) {
      /* Enviar valor local de cs al proceso (rank + 1) % size */
      MPI_Send(&U, 1, MPI_INT, siguiente, 0, comunicadorCota);
      pendiente_retorno_cs = true;
      difundir_cs_local = false;
    } else if (status.MPI_SOURCE == rank && !difundir_cs_local) {
      pendiente_retorno_cs = false;
    } else { // origen mensaje == otro proceso
      /* Reenviar mensaje al proceso (rank + 1) % size */
      MPI_Send(&U, 1, MPI_INT, siguiente, 0, comunicadorCota);
    }
    /* Sondear si hay mensajes de cota superior pendientes */
    MPI_Iprobe(anterior, MPI_ANY_TAG, comunicadorCota, &hay_mensajes, &status);
  }
}
