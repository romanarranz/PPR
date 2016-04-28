#include <iostream>
#include <cstdio>       // exit
#include <cstdlib>      // rand
#include <fstream>
#include <string.h>
using namespace std;

void guardarTSP(int ** M, int NCIUDADES){
    string nombreArchivo = "input/tsp";
    nombreArchivo += to_string(NCIUDADES)+".1";
    ofstream archivo (nombreArchivo);
    if (archivo.is_open()){
        for(int i = 0; i<NCIUDADES; i++){
            for(int j = 0; j<NCIUDADES; j++)
                archivo << to_string(M[i][j]) << "\t";
            archivo << "\n";
        }
        archivo.close();
    }
    else
        cout << "No se puede abrir el archivo";
}

int main(int argc, char ** argv){

    if(argc != 2){
        cerr << "Error: la sintaxis es <" << argv[0] << "> <NCIUDADES>" << endl;
        exit(-1);
    }

    int NCIUDADES = atoi(argv[1]);
    if(NCIUDADES <= 0){
        cerr << "Error: el numero de ciudades tiene que ser mayor a 0" << endl;
        exit(-1);
    }

    /* initialize random seed: */
    srand (time(NULL));

    // Rellenar la matriz
    int ** M = new int* [NCIUDADES];
    for(int i = 0; i<NCIUDADES; i++){
        M[i] = new int[NCIUDADES];
        for(int j = 0; j<NCIUDADES; j++){
            // Rellenar la diagonal a 0
            if(j == i) M[i][j] = 0;
            else M[i][j] = rand() % (NCIUDADES*5) + 1;
        }
    }

    guardarTSP(M, NCIUDADES);

    return EXIT_SUCCESS;
}
