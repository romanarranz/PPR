#include "functions.h"

void DifusionCotaSuperior(){
	if (difundir_cs_local && !pendiente_retorno_cs)  {

        // Enviar valor local de cs al proceso (id+1)%P;
		MPI_Send(&U, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);

        pendiente_retorno_cs = true;
        difundir_cs_local = false;
    }

    // Sondear si hay mensajes de cota superior pendientes
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &status);

    while(flag) // mientras tengamos mensajes
    {
        // Recibir mensaje con valor de cota superior desde el proceso (id-1+P)%P
        // Actualizar valor local de cota superior
		MPI_Recv(&U, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &status);

        if (status.MPI_SOURCE == id && difundir_cs_local){
            // Enviar valor local de cs al proceso (id+1)%P;
			MPI_Send(&U, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);

			pendiente_retorno_cs = true;
            difundir_cs_local = false;
        }
        else if (status.MPI_SOURCE == id && !difundir_cs_local){
			pendiente_retorno_cs = false;
		}
		else { // origen mensaje = otro proceso
			// reenviar mensaje al proceso (id+1)%p;
			MPI_Send(&U, 1, MPI_INT, (id+1)%P, PETICION, COMM_EQUILIBRADO_CARGA);
		}

		// Sondear si hay mensajes de cota superior pendientes
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &status);
	}
}
