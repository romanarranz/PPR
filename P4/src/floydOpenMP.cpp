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
    int k, i, j, vikj, iGlobal, jGlobal;
    int sqrtP = sqrt(P);
    int tamBloque = N/sqrtP;
    int chunk = 1;//tamBloque/P;


    int iLocalInicio, iLocalFinal;
    int jLocalInicio, jLocalFinal;
    iLocalInicio = iLocalFinal = jLocalInicio = jLocalFinal = 0;

    int idProcesoBloqueK;
    int indicePartidaFilaK;


    int * filak = new int[tamBloque];
    int * columnak = new int[tamBloque];
    for(i = 0; i<tamBloque; i++){
        filak[i] = 0;
        columnak[i] = 0;
    }

    printf("Hay un total de %u hebras, cada se encarga de %u filas consecutivas\n", P, chunk );
    double t1 = omp_get_wtime();
    for(k = 0; k<N; k++){

        #pragma omp parallel shared(k,M,chunk,tamBloque,filak,columnak) private(i,j,vikj,iLocalInicio,iLocalFinal,jLocalInicio,jLocalFinal)
        {
            iLocalInicio = (omp_get_thread_num()/sqrtP) * tamBloque;
            iLocalFinal = ((omp_get_thread_num()/sqrtP)+1) * tamBloque;

            jLocalInicio = (omp_get_thread_num()%sqrtP) * tamBloque;
            jLocalFinal = ((omp_get_thread_num()%sqrtP)+1) * tamBloque;

            idProcesoBloqueK = k / tamBloque;
            indicePartidaFilaK = k % tamBloque;

            if (k >= iLocalInicio && k < iLocalFinal){
                int ik = (k*N) + jLocalInicio;
                copy(&M[ik], &M[ik] + tamBloque, filak);
                // imprimir valores
            }

            if (k >= jLocalInicio && k < jLocalFinal){
                for (i = 0; i < tamBloque; i++){
                    int kj = ((iLocalInicio + i )*N) + k;
                    columnak[i] = M[kj];
                    // imprimir valores
                }
            }

            #pragma omp barrier
            printf("k = %u \n\tT%u -> iStart: %u, iEnd:%u || jStart: %u, jEnd: %u\n", k, omp_get_thread_num(), iLocalInicio, iLocalFinal, jLocalInicio, jLocalFinal);

            #pragma omp for schedule(static, chunk)
            for(i = iLocalInicio; i<iLocalFinal; i++){
                iGlobal = iLocalInicio + i;

                // imprimir iGlobal
                for(j = jLocalInicio; j<jLocalFinal; j++){
                    jGlobal = jLocalInicio + j;

                    if (iGlobal != jGlobal && iGlobal != k && jGlobal != k)
                    {
                        int ij = (iGlobal*N) + j;
                        vikj = columnak[i] + filak[j];
                        vikj = min(vikj, M[ij]);
                        M[ij] = vikj;
                    }
                }
            }
        }
    }
    double t2 = omp_get_wtime();

    return (t2-t1);
}
