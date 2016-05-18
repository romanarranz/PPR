extern "C"{
    void floyd1DGPU(int *M, int N, int numBloques, int numThreadsBloque);
    void floyd2DGPU(int *M, int N, int numBloques, int numThreadsBloque);
}
