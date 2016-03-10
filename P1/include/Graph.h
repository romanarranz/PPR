#ifndef GRAPH_H
#define GRAPH_H

#define INF 1000000

#include <vector>

using namespace std;

class Graph
{
	private:
		int *A;
	
	public:
        Graph();
        ~Graph();
        int vertices;
	   	void fija_nverts(const int &verts);
	   	void inserta_arista(const int &ptA, const int &ptB, const int &edge);
	   	int arista(const int ptA,const int ptB);
       	void imprime();
        void lee(char *filename);
        int * getPtrMatriz();
        vector<int> getFilaK(int k);
};

#endif