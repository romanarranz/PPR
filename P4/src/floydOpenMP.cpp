#include <omp.h>
#include <stdio.h>
#include <algorithm>    // std::min
using std::min;
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
