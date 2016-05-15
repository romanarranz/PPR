#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

// CUDA runtime
#include <cuda_runtime.h>


using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{


if (argc != 2) 
	{
	 cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	return(-1);
	}

Graph G;
G.lee(argv[1]);		// Read the Graph
//cout << "EL Grafo de entrada es:"<<endl;
//G.imprime();

int nverts=G.vertices;
double  t1=clock();
// BUCLE PPAL DEL ALGORITMO
for(int k=0;k<nverts;k++)
   for(int i=0;i<nverts;i++)
     for(int j=0;j<nverts;j++)
      if (i!=j && i!=k && j!=k) 
        {
         int vikj=min(G.arista(i,k)+G.arista(k,j),G.arista(i,j));
         G.inserta_arista(i,j,vikj);   
        }
  double t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;
  cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl<<endl;
  //G.imprime();
  cout<< "Tiempo gastado= "<<t2<<endl<<endl;
}



