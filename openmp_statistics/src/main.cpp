#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;

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

    int chunk = N/(2 * omp_get_num_threads());
    int i;

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,chunk) shared(v,N,chunk) private(i) default(none)
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

int main(){

    int option = 0;
    double t;

    const unsigned int N = 10000;
    const unsigned int k = 8;

    const unsigned int memSize = N * sizeof(int);
    int * V = (int *) malloc(memSize);
    int * A = (int *) malloc(memSize);

    for(int i = 0; i<N; i++){
        V[i] = 1 + i;
        A[i] = 1 + i;
    }

    do {
        cout << "****** Testing OpenMP Statistics ******" << endl;
        cout << "Execute very computational cost function with differents options" << endl;
        cout << "Option 1 => schedule static ciclic" << endl;
        cout << "Option 2 => schedule static blocks" << endl;
        cout << "Option 3 => schedule dynamic" << endl;
        cout << "Option 4 => schedule guided" << endl << endl;

        cout << "Select testing method: ";
        cin >> option;
        cout << endl;

        cout << "You select the option " << option << endl;
        switch(option){
            case 1:
                t = schedule_static_ciclic(V, N, k);
                break;

            case 2:
                t = schedule_static_blocks(V, N, k);
                break;

            case 3:
                t = schedule_dynamic(V, N, k);
                break;

            case 4:
                t = schedule_guided(V, N, k);
                break;
            default:
                cout << "No option selected" << endl;
                break;
        }
    }
    while(option == 0 || option > 4);

    // Test results
    for(int i = 0; i<N; i++){
        A[i] = f(A[i]);
    }

    bool ok = true;
    for(int i = 0; i<N; i++){
        if(A[i] != V[i])
            ok = false;
    }

    if(!ok)
        cout << "With ERR" << endl;
    else
        cout << "ALL OKS" << endl;

    cout << "Calc time: " << t << endl;

    free(V);
    free(A);
}
