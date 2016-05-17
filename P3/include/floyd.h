extern "C"{
    void floyd1DGPU(int *M, Graph g, int N, int numBloques, int numThreadsBloque);
    void floyd2DGPU(int *M, Graph g, int N, int numBloques, int numThreadsBloque);
}
