#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"


#define blocksize 256

using namespace std;

//**************************************************************************
// Kernel to update the Matrix at k-th iteration
__global__ void floyd_kernel(int * M, const int nverts, const int k)
  { 
//*******************************************************************

    
    
//*******************************************************************
       
  }



int main (int argc, char *argv[])

{

  

if (argc != 2) 
	{
	 cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	return(-1);
	}
	


    //Get GPU information
    int devID;
    cudaDeviceProp props;
    cudaError_t err;
    err=cudaGetDevice(&devID);
    if (err!=cudaSuccess) {cout<<"ERRORRR"<<endl;}
    
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

     
Graph G;
G.lee(argv[1]);		// Read the Graph

//cout << "EL Grafo de entrada es:"<<endl;
//G.imprime();

const int nverts=G.vertices;
const int niters=nverts;

const int nverts2=nverts*nverts;

int *c_Out_M=new int[nverts2]; 

int size=nverts2*sizeof(int);

int * d_In_M=NULL;




// GPU phase

err=cudaMalloc((void **) &d_In_M, size); 
if (err!=cudaSuccess) {cout<<"ERROR: Bad Allocation in Device Memory"<<endl;}

double  t1=clock();

err=cudaMemcpy(d_In_M,G.Get_Matrix(),size,cudaMemcpyHostToDevice);
if (err!=cudaSuccess) {cout<<"ERROR: COPY MATRIX TO DEVICE"<<endl;}

for(int k=0;k<niters;k++)
  {
  
//*******************************************************************

//Kernel Launch 


 
//*******************************************************************  
  
  err = cudaGetLastError();
    
  if (err != cudaSuccess)
     {
       fprintf(stderr, "Failed to launch kernel!\n");
       exit(EXIT_FAILURE);
     }
	    
  }


cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);


double Tgpu=clock();
Tgpu=(Tgpu-t1)/CLOCKS_PER_SEC;
cout<< "Tiempo gastado GPU= "<<Tgpu<<endl<<endl;



// CPU phase

t1=clock();
// BUCLE PPAL DEL ALGORITMO

for(int k=0;k<niters;k++)
   for(int i=0;i<nverts;i++)
     for(int j=0;j<nverts;j++)
      if (i!=j && i!=k && j!=k) 
        {
         int vikj=min(G.arista(i,k)+G.arista(k,j),G.arista(i,j));
         G.inserta_arista(i,j,vikj);   
        }
	
  double t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;
//  cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl<<endl;
//  G.imprime();
  cout<< "Tiempo gastado CPU= "<<t2<<endl<<endl;
  cout<< "Ganancia= "<<t2/Tgpu<<endl;
   
  for(int i=0;i<nverts;i++)
    for(int j=0;j<nverts;j++)
       if (abs(c_Out_M[i*nverts+j]-G.arista(i,j))>0) 
         cout <<"Error ("<<i<<","<<j<<")   "
	       <<c_Out_M[i*nverts+j]<<"..."<<G.arista(i,j)<<endl;
	
}

