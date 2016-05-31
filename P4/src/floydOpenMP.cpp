#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>    // std::min
using std::min;
using std::copy;
#include "floyd.h"

double floyd1DOpenMP(int * M, const int N, const int P){

    int k, i, j, vikj;
    int chunk = N/P;

    omp_set_dynamic(0);
    // lo optimo seria poner omp_get_num_procs() para aprovechar los recursos hw reales
    omp_set_num_threads(P);

    printf("Hay un total de %u hebras, cada se encarga de %u filas consecutivas\n", P, chunk );
    double t1 = omp_get_wtime();
	for(k = 0; k<N; k++){
        // poner shared k lo hacemos para que todas las hebras conozcan la fila k, equivalente al mpi_broadcast k
        #pragma omp parallel shared(M,k,chunk) private(i,j,vikj)
        {
            #pragma omp for schedule(static, chunk)
    		for(i = 0; i<N; i++){
                int ik = i*N + k;
                //printf("k:%u\n\tT%u -> fila %u\n", k, omp_get_thread_num(),i );
                for(j = 0; j<N; j++){
                    if(i!=j && i!=k && j!=k){
                        int kj = k*N + j;
                        int ij = i*N + j;
                        vikj = min(M[ik] + M[kj], M[ij]);
                        M[ij] = vikj;
                    }
                }
            }
    	}
    }
    double t2 = omp_get_wtime();

    return (t2-t1);
}

double floyd2DOpenMP(int * M, const int N, const int P){
    int k, i, j, vikj;
    int sqrtP = sqrt(P);
    int tamBloque = N/sqrtP;
    int chunk = 1;//tamBloque/P;

    omp_set_dynamic(0);
    omp_set_num_threads(P);

    int iLocalInicio, iLocalFinal;
    int jLocalInicio, jLocalFinal;
    iLocalInicio = iLocalFinal = jLocalInicio = jLocalFinal = 0;

    // filak y columnak declaradas en el stack
    int filak[N];
    int columnak[N];

    for(i = 0; i<N; i++){
        filak[i] = 0;
        columnak[i] = 0;
    }

    printf("Hay un total de %u hebras, cada se encarga de %u filas consecutivas\n", P, chunk );
    double t1 = omp_get_wtime();

    // ponemos la construccion del bloque paralelo aqui para ahorrar sincronizacion de hebras
    #pragma omp parallel shared(M,chunk,tamBloque,columnak,filak) private(k,i,j,vikj,iLocalInicio,iLocalFinal,jLocalInicio,jLocalFinal)
    {
        int tidDSqrt = omp_get_thread_num()/sqrtP;
        int tidMSqrt = omp_get_thread_num()%sqrtP;

        iLocalInicio = tidDSqrt * tamBloque;
        iLocalFinal = (tidDSqrt+1) * tamBloque;

        jLocalInicio = tidMSqrt * tamBloque;
        jLocalFinal = (tidMSqrt+1) * tamBloque;

        for(k = 0; k<N; k++){
            #pragma omp barrier

            for(i = 0; i<N; i++){
                filak[i] = M[ k * N + i];
                columnak[i] = M[i * N + k];
            }

            // printf("k = %u \n\tT%u -> iStart: %u, iEnd:%u || jStart: %u, jEnd: %u\n", k, omp_get_thread_num(), iLocalInicio, iLocalFinal, jLocalInicio, jLocalFinal);

            for(i = iLocalInicio; i<iLocalFinal; i++){
                int in = i * N;
                for(j = jLocalInicio; j<jLocalFinal; j++){
                    // printf("k=%u, T%u => i:%u , j:%u\n", k, omp_get_thread_num(), i, j );
                    if (i != j && i != k && j != k){
                        int ij = in + j;
                        vikj = columnak[i] + filak[j];
                        vikj = min(vikj, M[ij]);
                        M[ij] = vikj;
                    }
                }
            }
        }
    } // end parallel
    double t2 = omp_get_wtime();

    return (t2-t1);
}
