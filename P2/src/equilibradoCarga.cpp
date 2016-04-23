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

tNodo *solucionLocal = new tNodo();

void EquilibrarCarga(tPila * pila, bool *fin, tNodo *solucion){
	#if !DEBUG_EQUILIBRADO
		cout.setstate(ios_base::failbit);
	#else
		cout.clear();
	#endif

	color = BLANCO;

	if(pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
		cout << "[EQ] " << id << " NO tiene nodos " << endl;
		#if !DEBUG_EQ_SLEEP
			sleep(1);
		#endif

		//ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
		MPI_Send(&id, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
		cout << "[EQ] " << id << " tiene la pila vacia y le manda peticion a " << siguiente << endl;
		#if DEBUG_EQ_SLEEP
			sleep(1);
		#endif

		while(pila->vacia() && !(*fin)){

			// ESPERAR MENSAJE DE OTRO PROCESO --> BLOQUEANTE
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &status);
			cout << "[EQ] " << id << " SONDEO" << endl;
			#if DEBUG_EQ_SLEEP
				sleep(1);
			#endif

			switch(status.MPI_TAG){
				case NODOS:		// al proceso le ha llegado un mensaje que tiene nodos cedidos por otro proceso
					estado = ACTIVO;
					cout << "[EQ] " << id << " recibio un mensaje de nodos" << endl;
					#if DEBUG_EQ_SLEEP
						sleep(1);
					#endif

					int cantidadNodos;
					// OBTENER LA CANTIDAD DE NODOS QUE SE HAN DONADO
					MPI_Get_count(&status, MPI_INT, &cantidadNodos);

					//RECIBIR NODOS DEL PROCESO DONANTE EN LA PILA
					MPI_Recv(&pila->nodos[0], cantidadNodos, MPI_INT, MPI_ANY_SOURCE, NODOS, COMM_EQUILIBRADO_CARGA, &status);
					pila->tope = cantidadNodos;
					cout << "[EQ] " << id << " recibio nodos, tiene ahora " << pila->tope << " nodos " << endl;
					#if DEBUG_EQ_SLEEP
						sleep(1);
					#endif

					break;

				case PETICION: 	// peticion de trabajo

					// RECIBIR MENSAJE DE PETICION DE TRABAJO DEL PROCESO ANTERIOR QUE HIZO EL ENVIO
					MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);
					cout << "[EQ] " << id << " recibio un mensaje de peticion de " << anterior << endl;
					#if DEBUG_EQ_SLEEP
						sleep(1);
					#endif

					// ENVIO EL MENSAJE AL SIGUIENTE
					MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
					cout << "[EQ] " << id << " le pasa la peticion a " << siguiente << endl;
					#if DEBUG_EQ_SLEEP
						sleep(1);
					#endif

					if(solicitante == id){ // el mensaje dio la vuelta al anillo de procesos y ha vuelto al solicitante original
						// Se ha agotado la pila local y la peticion que hicimos de trabajo nos volvió será porque ya no queda trabajo
						estado = PASIVO;
						cout << "[FIN] " << id << " la peticion le ha llegado de vuelta y se queda PASIVO" << endl;
						#if DEBUG_EQ_SLEEP
							sleep(1);
						#endif

						// Cuando un proceso ademas tiene el testigo se reinicia la deteccion de fin enviandolo al proceso anterior
						if(token_presente){
							if(id == 0){
								color_token = BLANCO;
								cout << "[FIN] " << id << " ademas tiene el token:" << color_token << endl;
								#if DEBUG_EQ_SLEEP
									sleep(1);
								#endif
							}
							else{
								color_token = color;
								cout << "[FIN] " << id << " ademas tiene el token:" << color_token << endl;
								#if DEBUG_EQ_SLEEP
									sleep(1);
								#endif
							}

							// como el proceso esta pasivo le mandamos el token al anterior
							MPI_Send(&color_token, 1, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
							token_presente = false;
							color = BLANCO;
							cout << "[FIN] " << id << " PASIVO le manda el token a " << anterior << " y queda LIMPIO el proceso" << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif
						}
					}

					break;

				case TOKEN: // LE HA LLEGADO UN TOKEN AL PROCESO
					token_presente = true;
					cout << "[FIN] " << id;

					// si el estado del proceso es pasivo
					if(estado == PASIVO){
						cout << " PASIVO recibio el token:" << color_token << endl;
						#if DEBUG_EQ_SLEEP
							sleep(1);
						#endif

						// y ademas el token ha dado la vuelta hasta el 0 que lo inicio y permanece limpio llegamos al fin
						if(id == 0 && color == BLANCO && color_token == BLANCO){
							*fin = true;
							cout << "[FIN] el token ha dado la vuelta completa hasta el 0 -> FIN VERDADERO" << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif

							// [PUBLICAR SOLUCION] RECIBIR EL MENSAJE DE FIN DEL ANTERIOR
							MPI_Recv(&solucionLocal->datos[0], solucion->size(), MPI_INT, anterior, FIN, COMM_EQUILIBRADO_CARGA, &status);
							cout << "[SOL] solucion recibida, se la pasamos a " << siguiente << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif

							// si la solucion local que tiene el proceso es mejor que la solucion final la reemplazamos
							if(solucionLocal->ci() < solucion->ci()){
								CopiaNodo(solucionLocal, solucion);
							}
							delete solucionLocal;

							// [PUBLICAR SOLUCION] ENVIAR EL MENSAJE DE FIN AL SIGUIENTE
							MPI_Send(&solucion->datos[0], solucion->size(), MPI_INT, siguiente, FIN, COMM_EQUILIBRADO_CARGA);

						}
						else {
							if(id == 0){
								color_token = BLANCO;
							}
							else if(color == NEGRO){
								color_token = NEGRO;
							}

							// ENVIAR TOKEN AL ANTERIOR
							MPI_Send(&color_token, 1, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
							color = BLANCO;
							token_presente = false;
							cout << "[FIN] " << id << " tiene que seguir enviando el token a " << anterior << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif
						}
					}

					break;

				case FIN: 	// TRABAJO AGOTADO PARA TODOS LOS PROCESOS
					*fin = true;

					// [PUBLICAR SOLUCION] RECIBIR EL MENSAJE DE FIN DEL ANTERIOR
					MPI_Recv(&solucionLocal->datos[0], solucion->size(), MPI_INT, anterior, FIN, COMM_EQUILIBRADO_CARGA, &status);
					// si la solucion local que tiene el proceso es mejor que la solucion final la reemplazamos
					if(solucionLocal->ci() < solucion->ci()){
						CopiaNodo(solucionLocal, solucion);
					}
					delete solucionLocal;

					// [PUBLICAR SOLUCION] ENVIAR EL MENSAJE DE FIN AL SIGUIENTE
					MPI_Send(&solucion->datos[0], solucion->size(), MPI_INT, siguiente, FIN, COMM_EQUILIBRADO_CARGA);
					cout << "[SOL] solucion recibida, se la pasamos a " << siguiente << endl;
					#if DEBUG_EQ_SLEEP
						sleep(1);
					#endif

					break;
			}
		}

		// El proceso tiene nodos para trabajar
		if(!(*fin)){
			cout << "[EQ] " << id << " SI tiene nodos para trabajar \n";
			#if DEBUG_EQ_SLEEP
				sleep(1);
			#endif

			// sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
			cout << "[EQ] " << id << " SONDEO" << endl;
			#if DEBUG_EQ_SLEEP
				sleep(1);
			#endif

			while(hay_mensajes){ // atiende peticiones mientras haya mensajes
				switch(status.MPI_TAG){
					case PETICION:
						// RECIBIR MENSAJE DE PETICION DE TRABAJO, le ha llegado al proceso una demanda de nodos
						MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

						if(pila->tamanio() > 1){  // si al menos tenemos 2 nodos en la pila, podemos ceder uno
							tPila pila2;

							// DIVIDIR LA PILA
							pila->divide(pila2);
							cout << "[EQ] " << id  << " tiene " << pila->tope << " nodos " << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif

							// ENVIAR NODOS AL PROCESO SOLICITANTE
							MPI_Send(&pila2.nodos[0], pila2.tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
							delete &pila2;
							cout << "[EQ] " << id  << " envia nodos al solicitante " << solicitante << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif


							if(id < solicitante){
								color = NEGRO;
								cout << "[FIN] " << id << " queda manchado porque le cede nodos al proceso " << solicitante << endl;
								#if DEBUG_EQ_SLEEP
									sleep(1);
								#endif
							}

						}
						else { // no tenemos suficientes nodos como para ceder
							// PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
							MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
							cout << "[EQ] " << id << " no tiene suficientes nodos para pasarselos a " << solicitante << " asi que se los pasa a " << siguiente << endl;
							#if DEBUG_EQ_SLEEP
								sleep(1);
							#endif
						}

						// sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
						MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
						cout << "[EQ] " << id << " SONDEO" << endl;
						#if DEBUG_EQ_SLEEP
							sleep(1);
						#endif

						break;

					case TOKEN: // al proceso que aun tiene nodos para trabajar le ha llegado el token
						// recibimos el token del siguiente
						MPI_Recv(&color_token, 1, MPI_INT, siguiente, TOKEN, COMM_EQUILIBRADO_CARGA, &status);
						token_presente = true;
						cout << "[FIN] " << id << " recibe el token de " << siguiente << " pero aun le quedan nodos en la pila " << endl;
						#if DEBUG_EQ_SLEEP
							sleep(1);
						#endif

						break;
				}

				// sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);
				cout << "[EQ] " << id << " SONDEO" << endl;
				#if DEBUG_EQ_SLEEP
					sleep(1);
				#endif
			} // fin while hay mensajes
		} // fin if !fin
	} // fin if pila->vacia
}
