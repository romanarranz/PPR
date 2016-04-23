#include "functions.h"

// Tipos de mensajes que se envian los procesos
const int PETICION = 0;
const int NODOS = 1;
const int TOKEN = 2;
const int FIN = 3;

// Estados en los que se puede encontrar un proceso
const int ACTIVO = 0;
const int PASIVO = 1;

// Colores que pueden tener tanto los procesos como el token
const int BLANCO = 0;
const int NEGRO = 1;

void EquilibrarCarga(tPila * pila, bool *fin){
	#if !DEBUG_EQUILIBRADO
		cout.setstate(ios_base::failbit);
	#else
		cout.clear();
	#endif

	if(pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
		cout << "[EQ] " << id << " NO tiene nodos " << endl;
		sleep(1);

		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		MPI_Send(&id, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
		cout << "[EQ] " << id << " tiene la pila vacia y le manda peticion a " << siguiente << endl;
		sleep(1);

		while(pila->vacia() && !(*fin)){

			// ESPERAR MENSAJE DE OTRO PROCESO --> BLOQUEANTE
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &status);

			cout << "[EQ] " << id << " SONDEO" << endl;
			sleep(1);

			switch(status.MPI_TAG){
				case PETICION: 	// peticion de trabajo

					//RECIBIR MENSAJE DE PETICION DE TRABAJO DEL PROCESO ANTERIOR QUE HIZO EL ENVIO
					MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

					cout << "[EQ] " << id << " recibio un mensaje de peticion de " << anterior << endl;
					sleep(1);

					if(solicitante == id){ // el mensaje dio la vuelta al anillo de procesos y ha vuelto al solicitante original
						// Si se ha agotado la pila local y la peticion que hicimos de trabajo nos volvió será porque ya no queda trabajo
						estado = PASIVO;

						//REENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
						cout << "[EQ] " << id << " le ha llegado la peticion de vuelta, lo envia a " << siguiente << endl;
						sleep(1);

						// Cuando un proceso ademas tiene el testigo se reinicia la deteccion de fin enviandolo al proceso anterior
						if(token_presente){
							MPI_Send(&token, 1, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
						}
					}
					else{ // le ha llegado un mensaje de peticion al proceso, pero como tiene su pila vacia entonces reenvia el mensaje al sig
						// PASAR LA PETICION AL PROCESO (id+1)%P
						MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
						cout << "[EQ] " << id << " le ha llegado una peticion de " << solicitante << ", se la pasa a " << siguiente << endl;
					}
					break;

				case NODOS:		// al proceso le ha llegado un mensaje que tiene nodos cedidos por otro proceso
					estado = ACTIVO;
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

				case TOKEN: // LE HA LLEGADO UN TOKEN AL PROCESO
					token_presente = true;
					if(estado == PASIVO){
						if(id == 0 && color == BLANCO && color_token == BLANCO){
							*fin = true;
						}
						else {
							if(id == 0){
								color_token = BLANCO;
							}
							else if(color == NEGRO){
								color_token = NEGRO;
							}

							// ENVIAR TOKEN A P(id-1)
							MPI_Send(&token, 1, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
							color = BLANCO;
							token_presente = false;
						}
					}

					break;

				case FIN: 	// TRABAJO AGOTADO PARA TODOS LOS PROCESOS
					estado = PASIVO;

					if(token_presente){
						if(id == 0){
							color_token = BLANCO;
						}
						else if(color == NEGRO){
							color_token = NEGRO;
						}

						// ENVIAR TOKEN A P(id-1)
						MPI_Send(&token, 1, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
						color = BLANCO;
						token_presente = false;
					}

					break;
			}
		}

		// El proceso tiene nodos para trabajar
		if(!(*fin)){
			cout << "[EQ] " << id << " SI tiene nodos para trabajar \n";
			sleep(1);

			// sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
			cout << "[EQ] " << id << " SONDEO" << endl;
			sleep(1);

			while(hay_mensajes){ // atiende peticiones mientras haya mensajes

				// RECIBIR MENSAJE DE PETICION DE TRABAJO, le ha llegado al proceso una demanda de nodos
				MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

				if(pila->tamanio() > 1){  // si al menos tenemos 2 nodos en la pila, podemos ceder uno
					tPila pila2;

					// DIVIDIR LA PILA
					pila->divide(pila2);
					cout << "[EQ] " << id  << " tiene " << pila->tope << " nodos " << endl;
					sleep(1);

					// ENVIAR NODOS AL PROCESO SOLICITANTE
					MPI_Send(&pila2.nodos[0], pila2.tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
					cout << "[EQ] " << id  << " envia nodos al solicitante " << solicitante << endl;
					sleep(1);
				}
				else { // no tenemos suficientes nodos como para ceder
					// PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
					MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
				}

				// sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
				cout << "[EQ] " << id << " SONDEO" << endl;
				sleep(1);

			}
		}
	}
}
