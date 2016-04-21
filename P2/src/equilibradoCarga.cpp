#include "functions.h"

const int PETICION = 0;
const int NODOS = 1;

void EquilibrarCarga(tPila * pila, bool *fin){

	if(pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
		cout << " pila vacia " << id << endl;
		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		MPI_Send(&id, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);

		while(pila->vacia() && !(*fin)){

			//ESPERAR MENSAJE DE OTRO PROCESO
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &status);

			switch(status.MPI_TAG){
				case 0: 	// peticion de trabajo

					//RECIBIR MENSAJE DE PETICION DE TRABAJO
					MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &status);

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
					MPI_Get_count(&status, MPI_INT, &cantidadNodos);

					//RECIBIR NODOS DEL PROCESO DONANTE EN LA PILA
					MPI_Recv(&pila->nodos[pila->tope], cantidadNodos, MPI_INT, MPI_ANY_SOURCE, NODOS, COMM_EQUILIBRADO_CARGA, &status);
					pila->tope = cantidadNodos;
					break;
			}

			*fin = true;
		}

		// El proceso tiene nodos para trabajar
		if(!(*fin)){
			tPila pila2;

			// sondear si hay mensajes pendientes de otros procesos
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &status);

			while(flag){ // atiende peticiones mientras haya mensajes

				// RECIBIR MENSAJE DE PETICION DE TRABAJO
				MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &status);

				if(pila->tope > 1){  // hay suficientes nodos en la pila para ceder

					// DIVIDIR LA PILA
					pila->divide(pila2);

					// ENVIAR NODOS AL PROCESO SOLICITANTE
					MPI_Send(&pila2.nodos[0], pila2.tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
				}
				else {
					// PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
					MPI_Send(&solicitante, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);
				}

				// sondear si hay mensajes pendientes de otros procesos
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &status);
			}
			*fin = true;
		}
	}
}
