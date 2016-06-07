#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sstream>
#include <fstream>

#include "schedules.h"

using namespace std;

void guardarArchivo(string outputFile, int n, double t){
    ofstream archivo (outputFile.c_str(), ios_base::app | std::ios_base::out);
    if (archivo.is_open()){
        stringstream ns, ts;
        ns << n;
        ts << t;
        string input =  ns.str() + "\t" + ts.str() + "\n";
        archivo << input;
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

int main(int argc, char ** argv){

    if(argc < 2){
        cerr << "Error: " << argv[0] << " size" << endl;
        exit(-1);
    }

    int option = 0;
    double t;

    const unsigned int N = atoi(argv[1]);
    const unsigned int k = 8;

    const unsigned int memSize = N * sizeof(int);
    int * V = (int *) malloc(memSize);
    int * A = (int *) malloc(memSize);

    for(int i = 0; i<N; i++){
        V[i] = 1 + i;
        A[i] = 1 + i;
    }

    string fileName = "";
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
                fileName = "staticCiclic.dat";
                break;

            case 2:
                t = schedule_static_blocks(V, N, k);
                fileName = "staticBlocks.dat";
                break;

            case 3:
                t = schedule_dynamic(V, N, k);
                fileName = "dynamic.dat";
                break;

            case 4:
                t = schedule_guided(V, N, k);
                fileName = "guided.dat";
                break;
            default:
                cout << "No option selected" << endl;
                fileName = "guided.dat";
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

    guardarArchivo("output/"+fileName, N, t);

    free(V);
    free(A);
}
