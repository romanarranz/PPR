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
    int chunk = 4;

    double t1 = omp_get_wtime();
    #pragma omp parallel shared(M) private(i,j,k, vikj) default(none)
	for(k = 0; k<N; k++){
        printf("k: %u\n", k);
        #pragma for schedule(static, chunk)
		for(i = 0; i<N; i++){
			for(j = 0; j<N; j++){
                vikj = min(M[i*N + k] + M[k*N + j], M[i*N + j]);
                M[i*N + j] = vikj;
                printf("\tT%u -> M[%u,%u] = %u\n", omp_get_thread_num(), i, j, vikj);
            }
        }
	}
    double t2 = omp_get_wtime();

    return (t2-t1);
}
