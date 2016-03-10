#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{
MPI::Init(argc,argv);

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

double t=MPI::Wtime();
// BUCLE PPAL DEL ALGORITMO
int i,j,k,vikj;
for(k=0;k<nverts;k++)
  {
   for(i=0;i<nverts;i++)
     for(j=0;j<nverts;j++)
      if (i!=j && i!=k && j!=k) 
        {
         vikj=G.arista(i,k)+G.arista(k,j);
         vikj=min(vikj,G.arista(i,j));
         G.inserta_arista(i,j,vikj);   
        }
  }
  t=MPI::Wtime()-t;
  MPI::Finalize();

 
  cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl<<endl;
  G.imprime();
  cout<< "Tiempo gastado= "<<t<<endl<<endl;


}



