#include "schedules.h"

#include <omp.h>
#include <math.h>

int f(int i){
    int res = 0;
    for(int k = 0; k<i; k++)
        res += k * ( (int) sqrt(k) + 4 % 11);

    return res;
}

double schedule_static_ciclic(int * v, int N, const int k){
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    int chunk = 1;
    int i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static,chunk) shared(v,N,chunk) private(i) default(none)
    for(i = 0; i<N; i++){
        v[i] = f(v[i]);
    }
    double t2 = omp_get_wtime();

    return t2 - t1;
}

double schedule_static_blocks(int * v, int N, const int k){
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    int chunk = N/(2 * omp_get_num_threads());
    int i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static,chunk) shared(v,N,chunk) private(i) default(none)
    for(i = 0; i<N; i++){
        v[i] = f(v[i]);
    }
    double t2 = omp_get_wtime();

    return t2 - t1;
}

double schedule_dynamic(int * v, int N, const int k){
    omp_set_dynamic(1);
    omp_set_num_threads(omp_get_num_procs());

    int i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) shared(v,N) private(i) default(none)
    for(int i = 0; i<N; i++){
        v[i] = f(v[i]);
    }
    double t2 = omp_get_wtime();

    return t2 - t1;
}

double schedule_guided(int * v, int N, const int k){
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    int i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(guided) shared(v,N) private(i) default(none)
    for(int i = 0; i<N; i++){
        v[i] = f(v[i]);
    }
    double t2 = omp_get_wtime();

    return t2 - t1;
}
