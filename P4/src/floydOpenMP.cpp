#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_set_num_threads(4);
#endif
#include <stdio.h>
#include <algorithm>    // std::min
using std::min;
#include "floyd.h"

double floyd1DOpenMP(int * M, const int N){

    int k, i, j, vikj;
    double t1 = omp_get_wtime();

	for(k = 0; k<N; k++){
        // poner shared k lo hacemos para que todas las hebras conozcan la fila k, equivalente al mpi_broadcast k
        #pragma omp parallel shared(M,k) private(i,j,vikj) default(none)
        {
            #pragma for schedule(static)
    		for(i = 0; i<N; i++){
                int ik = i*N + k;
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
