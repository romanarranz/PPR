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

void EquilibrarCarga(tPila *pila, bool *fin, tNodo *solucion)
{
    color = BLANCO;
    if (pila->vacia())  // el proceso no tiene trabajo: pide a otros procesos
    {
        //ENVIAR PETICION DE TRABAJO AL PROCESO (id+1)%P
        MPI_Send(&id, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
        while (pila->vacia() && !(*fin))
        {
            // ESPERAR MENSAJE DE OTRO PROCESO --> BLOQUEANTE
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &status);
            switch (status.MPI_TAG)
            {
                case NODOS:     // al proceso le ha llegado un mensaje que tiene nodos cedidos por otro proceso
                    estado = ACTIVO;
                    // OBTENER LA CANTIDAD DE NODOS QUE SE HAN DONADO
                    int cantidadNodos;
                    MPI_Get_count(&status, MPI_INT, &cantidadNodos);

                    //RECIBIR NODOS DEL PROCESO DONANTE EN LA PILA
                    MPI_Recv(&pila->nodos[0], cantidadNodos, MPI_INT, status.MPI_SOURCE, NODOS, COMM_EQUILIBRADO_CARGA, &status);
                    pila->tope = cantidadNodos;

                    break;

                case PETICION:  // peticion de trabajo
                    // RECIBIR MENSAJE DE PETICION DE TRABAJO DEL PROCESO ANTERIOR QUE HIZO EL ENVIO
                    MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);

                    // ENVIO EL MENSAJE AL SIGUIENTE
                    MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);

                    if (solicitante == id) // el mensaje dio la vuelta al anillo de procesos y ha vuelto al solicitante original
                    {
                        // Se ha agotado la pila local y la peticion que hicimos de trabajo nos volvió será porque ya no queda trabajo
                        estado = PASIVO;

                        // Cuando un proceso ademas tiene el testigo, se reinicia la deteccion de fin enviandolo al proceso anterior
                        if (token_presente)
                        {
                            if (id == 0)
                                color_token = BLANCO;
                            else
                                color_token = color;

                            // como el proceso esta pasivo le mandamos el token al anterior
                            MPI_Send(nullptr, 0, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
                            token_presente = false;
                            color = BLANCO;
                        }
                    }
                    break;
                case TOKEN: // LE HA LLEGADO UN TOKEN AL PROCESO

                    MPI_Recv(nullptr, 0, MPI_INT, siguiente, TOKEN, COMM_EQUILIBRADO_CARGA, &status);
                    token_presente = true;

                    // si el estado del proceso es pasivo
                    if (estado == PASIVO) {

                        // y ademas el token ha dado la vuelta hasta el 0 que lo inicio y permanece limpio llegamos al fin
                        if (id == 0 && color == BLANCO && color_token == BLANCO) {

                            *fin = true;

                            // [PUBLICAR SOLUCION] ENVIAR EL MENSAJE DE FIN AL SIGUIENTE
                            MPI_Send(&solucion->datos[0], solucion->size(), MPI_INT, siguiente, FIN, COMM_EQUILIBRADO_CARGA);

                            // [PUBLICAR SOLUCION] RECIBIR EL MENSAJE DE FIN DEL ANTERIOR
                            solucionLocal = new tNodo();
                            MPI_Recv(&solucionLocal->datos[0], solucion->size(), MPI_INT, anterior, FIN, COMM_EQUILIBRADO_CARGA, &status);

                            // si la solucion local que tiene el proceso es mejor que la solucion final la reemplazamos
                            if (solucionLocal->ci() < solucion->ci())
                                CopiaNodo(solucionLocal, solucion);

                            delete solucionLocal;
                        }
                        else {
                            if (id == 0)
                                color_token = BLANCO;
                            else
                                color_token = NEGRO;

                            // ENVIAR TOKEN AL ANTERIOR
                            MPI_Send(nullptr, 0, MPI_INT, anterior, TOKEN, COMM_EQUILIBRADO_CARGA);
                            color = BLANCO;
                            token_presente = false;
                        }
                    }
                    break;
                case FIN:   // TRABAJO AGOTADO PARA TODOS LOS PROCESOS

                    *fin = true;

                    // [PUBLICAR SOLUCION] RECIBIR EL MENSAJE DE FIN DEL ANTERIOR
                    solucionLocal = new tNodo();
                    MPI_Recv(&solucionLocal->datos[0], solucion->size(), MPI_INT, anterior, FIN, COMM_EQUILIBRADO_CARGA, &status);

                    // si la solucion local que tiene el proceso es mejor que la solucion final la reemplazamos
                    if (solucionLocal->ci() < solucion->ci())
                        CopiaNodo(solucionLocal, solucion);
                    delete solucionLocal;

                    // [PUBLICAR SOLUCION] ENVIAR EL MENSAJE DE FIN AL SIGUIENTE
                    MPI_Send(&solucion->datos[0], solucion->size(), MPI_INT, siguiente, FIN, COMM_EQUILIBRADO_CARGA);

                    break;
            }
        }
    } // fin if pila->vacia

    // El proceso tiene nodos para trabajar
    if (!(*fin))
    {
        // sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);

        while (hay_mensajes) // atiende peticiones mientras haya mensajes
        {
            switch (status.MPI_TAG)
            {
                case PETICION:
                    // RECIBIR MENSAJE DE PETICION DE TRABAJO, le ha llegado al proceso una demanda de nodos
                    MPI_Recv(&solicitante, 1, MPI_INT, anterior, PETICION, COMM_EQUILIBRADO_CARGA, &status);
                    if (pila->tamanio() > 1)  // si al menos tenemos 2 nodos en la pila, podemos ceder uno
                    {
                        // DIVIDIR LA PILA
                        pila2 = new tPila();
                        pila->divide(*pila2);

                        // ENVIAR NODOS AL PROCESO SOLICITANTE
                        MPI_Send(&pila2->nodos[0], pila2->tope, MPI_INT, solicitante, NODOS, COMM_EQUILIBRADO_CARGA);
                        delete pila2;

                        if (id < solicitante)
                            color = NEGRO;
                    }
                    else {  // no tenemos suficientes nodos como para ceder
                        // PASAR PETICION DE TRABAJO AL PROCESO (id+1)%P
                        MPI_Send(&solicitante, 1, MPI_INT, siguiente, PETICION, COMM_EQUILIBRADO_CARGA);
                    }

                    // sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
                    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);

                    break;
                case TOKEN: // al proceso que aun tiene nodos para trabajar le ha llegado el token
                    // recibimos el token del siguiente
                    MPI_Recv(nullptr, 0, MPI_INT, siguiente, TOKEN, COMM_EQUILIBRADO_CARGA, &status);
                    token_presente = true;

                    break;
            }

            // sondear si hay mensajes pendientes de otros procesos --> NO BLOQUEANTE
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &hay_mensajes, &status);

        } // fin while hay mensajes
    } // fin if !fin
}
