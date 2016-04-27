#include "functions.h"

void DifusionCotaSuperior(int *U, bool *nueva_U)
{
    difundir_cs_local = *nueva_U;
    if (difundir_cs_local && !pendiente_retorno_cs)
    {
        // Enviar valor local de cs al proceso (id+1)%P;
        MPI_Send(U, 1, MPI_INT, siguiente, id, COMM_DIFUSION_COTA);
        pendiente_retorno_cs = true;
        difundir_cs_local = false;
    }

    // Sondear si hay mensajes de cota superior pendientes
    MPI_Iprobe(anterior, MPI_ANY_TAG, COMM_DIFUSION_COTA, &hay_mensajes, &status);

    while (hay_mensajes)   // mientras tengamos mensajes
    {
        // Recibir mensaje con valor de cota superior desde el proceso (id-1+P)%P
        int cotaSup;
        MPI_Recv(&cotaSup, 1, MPI_INT, anterior, status.MPI_TAG, COMM_DIFUSION_COTA, &status);

        // Actualizar valor local de cota superior
        if (cotaSup < *U){
            *U = cotaSup;
            *nueva_U = true;
        }

        if (status.MPI_SOURCE == id && difundir_cs_local){
            // Enviar valor local de cs al proceso (id+1)%P;
            MPI_Send(U, 1, MPI_INT, siguiente, id, COMM_DIFUSION_COTA);
            pendiente_retorno_cs = true;
            difundir_cs_local = false;
        }
        else if (status.MPI_SOURCE == id && !difundir_cs_local)
            pendiente_retorno_cs = false;
        else    // origen mensaje = otro proceso
            // reenviar mensaje al proceso (id+1)%p;
            MPI_Send(U, 1, MPI_INT, siguiente, status.MPI_TAG, COMM_DIFUSION_COTA);

        // Sondear si hay mensajes de cota superior pendientes
        MPI_Iprobe(anterior, MPI_ANY_TAG, COMM_DIFUSION_COTA, &hay_mensajes, &status);
    }
}
