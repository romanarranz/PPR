#define SLEEP_TIME 1000000         // 1000000 microsegundos = 1 segundo

// depuracion de la cota superior
#define DEBUG_COTA false        // mensajes activos
#define DEBUG_CS_SLEEP false    // sleeps activos

#include "functions.h"

void DifusionCotaSuperior(int *U, bool *nueva_U)
{
    #if !DEBUG_COTA
    cout.setstate(ios_base::failbit);
    #else
    cout.clear();
    #endif
    difundir_cs_local = *nueva_U;
    if (difundir_cs_local && !pendiente_retorno_cs)
    {
        // Enviar valor local de cs al proceso (id+1)%P;
        MPI_Send(U, 1, MPI_INT, siguiente, id, COMM_DIFUSION_COTA);
        cout << "[CS] " << id << " va a difundir su cs " << *U << " y no esta pendiente de retorno -> " << siguiente << endl;
        #if DEBUG_CS_SLEEP
        usleep(SLEEP_TIME);
        #endif
        pendiente_retorno_cs = true;
        difundir_cs_local = false;
    }
    // Sondear si hay mensajes de cota superior pendientes
    MPI_Iprobe(anterior, MPI_ANY_TAG, COMM_DIFUSION_COTA, &hay_mensajes, &status);
    cout << "[CS] " << id << " SONDEO CS " << endl;
    #if DEBUG_CS_SLEEP
    usleep(SLEEP_TIME);
    #endif
    while (hay_mensajes)   // mientras tengamos mensajes
    {
        cout << "[CS] " << id << " tiene mensaje del sondeo " << endl;
        #if DEBUG_CS_SLEEP
        usleep(SLEEP_TIME);
        #endif
        int cotaSup;
        // Recibir mensaje con valor de cota superior desde el proceso (id-1+P)%P
        MPI_Recv(&cotaSup, 1, MPI_INT, anterior, status.MPI_TAG, COMM_DIFUSION_COTA, &status);
        // Actualizar valor local de cota superior
        if (cotaSup < *U)
        {
            *U = cotaSup;
            *nueva_U = true;
        }
        if (status.MPI_SOURCE == id && difundir_cs_local)
        {
            // Enviar valor local de cs al proceso (id+1)%P;
            MPI_Send(U, 1, MPI_INT, siguiente, id, COMM_DIFUSION_COTA);
            cout << "[CS] " << id << " ha dado la vuelta completa y le envia la cs a " << siguiente << endl;
            #if DEBUG_CS_SLEEP
            usleep(SLEEP_TIME);
            #endif
            pendiente_retorno_cs = true;
            difundir_cs_local = false;
        }
        else if (status.MPI_SOURCE == id && !difundir_cs_local)
        {
            pendiente_retorno_cs = false;
        }
        else   // origen mensaje = otro proceso
        {
            // reenviar mensaje al proceso (id+1)%p;
            MPI_Send(U, 1, MPI_INT, siguiente, status.MPI_TAG, COMM_DIFUSION_COTA);
        }
        // Sondear si hay mensajes de cota superior pendientes
        MPI_Iprobe(anterior, MPI_ANY_TAG, COMM_DIFUSION_COTA, &hay_mensajes, &status);
    }
}
