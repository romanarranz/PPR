#include "functions.h"

const int PETICION = 0;
const int NODOS = 1;

void EquilibrarCarga(tPila * pila, bool *fin){
	#if !DEBUG_EQUILIBRADO
		cout.setstate(ios_base::failbit);
	#endif

	if(pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
		cout << "[EQ] " << id << " NO tiene nodos " << endl;
		sleep(1);

		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		MPI_Send(&id, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
		cout << "[EQ] " << id << " tiene la pila vacia y le manda peticion a " << siguiente << endl;
		sleep(1);

		while(pila->vacia() && !(*fin)){

			//ESPERAR MENSAJE DE OTRO PROCESO
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &status);

			cout << "[EQ] " << id << " SONDEO" << endl;
			sleep(1);

			switch(status.MPI_TAG){
				case PETICION: 	// peticion de trabajo

					//RECIBIR MENSAJE DE PETICION DE TRABAJO DEL PROCESO ANTERIOR QUE HIZO EL ENVIO
					MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

					cout << "[EQ] " << id << " recibio un mensaje de peticion de " << anterior << endl;
					sleep(1);

					if(solicitante == id){ // peticion devuelta
						//REENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
						cout << "[EQ] " << id << " le ha llegado la peticion de vuelta, lo envia a " << siguiente << endl;
						sleep(1);

						//INICIAR DETECCION DE POSIBLE SITUACION DE FIN ??????????

					}
					else{ // peticion de otro proceso: la retransmite al siguiente
						//PASAR LA PETICION AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
						cout << "[EQ] " << id << " le ha llegado una peticion de " << solicitante << ", se la pasa a " << siguiente << endl;
					}
					break;

				case NODOS:		// resultado de una peticion de trabajo
					cout << "[EQ] " << id << " recibio un mensaje de nodos" << endl;
					sleep(1);

					int cantidadNodos;
					// OBTENER LA CANTIDAD DE NODOS QUE SE HAN DONADO
					MPI_Get_count(&status, MPI_INT, &cantidadNodos);

					//RECIBIR NODOS DEL PROCESO DONANTE EN LA PILA
					MPI_Recv(&pila->nodos[0], cantidadNodos, MPI_INT, MPI_ANY_SOURCE, NODOS, COMM_EQUILIBRADO_CARGA, &status);
					pila->tope = cantidadNodos;

					cout << "[EQ] " << id << " tiene ahora " << pila->tope << " nodos " << endl;
					sleep(1);

					break;
			}
		}

		// El proceso tiene nodos para trabajar
		if(!(*fin)){
			cout << "[EQ] " << id << " SI tiene nodos para trabajar \n";
			sleep(1);

			tPila pila2;

			// sondear si hay mensajes pendientes de otros procesos
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
			cout << "[EQ] " << id << " SONDEO" << endl;
			sleep(1);

			while(hay_mensajes){ // atiende peticiones mientras haya mensajes

				// RECIBIR MENSAJE DE PETICION DE TRABAJO
				MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

				if(pila->tamanio() > 1){  // hay suficientes nodos en la pila para ceder

					// DIVIDIR LA PILA
					pila->divide(pila2);
					cout << "[EQ] " << id  << " tiene " << pila->tope << " nodos " << endl;
					sleep(1);

					// ENVIAR NODOS AL PROCESO SOLICITANTE
					MPI_Send(&pila2.nodos[0], pila2.tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
					cout << "[EQ] " << id  << " envia nodos al solicitante " << solicitante << endl;
					sleep(1);
				}
				else {
					// PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
					MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
				}

				// sondear si hay mensajes pendientes de otros procesos
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
				cout << "[EQ] " << id << " SONDEO" << endl;
				sleep(1);

			}
		}
	}
}
