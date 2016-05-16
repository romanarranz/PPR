// Forward declarations
extern "C" {
    void initVectors(float *A, float *B, float *C, int N);
    void computeGPU(float *A, float *B, float *C, int N, int numBloques, int numThreadsBloque);
}
