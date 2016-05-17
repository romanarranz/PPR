extern "C" {
    void initMatrixes(float *A, float *B, float *C, int N);
    void matAddGPU(float *A, float *B, float *C, int N, int numBloques, int numThreadsBloque);
}
