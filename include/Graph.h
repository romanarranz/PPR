#ifndef GRAPH_H
#define GRAPH_H

#define INF 1000000

class Graph
{
	private:
		int *A;
	
	public:
        Graph();
        int vertices;
	   	void fija_nverts(const int &verts);
	   	void inserta_arista(const int &ptA, const int &ptB, const int &edge);
	   	int arista(const int ptA,const int ptB);
       	void imprime();
        void lee(char *filename);
};

#endif