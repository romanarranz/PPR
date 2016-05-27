#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define SIZE 1024
#define min(a,b) ((a)<(b)?(a):(b))
#define NTHREADS 4

/* OpenMP function defn */
void omp_set_num_threads(int num_threads);
int omp_get_thread_num(void);

/* Function definitions */
unsigned long isqrt(unsigned long x);
void initMatrix(int first_row, int num_rows, int first_col, int num_cols, int ** A);
int ** floyd(int N, int ** A);
int ** floyd2D(int num_rows, int first_row, int first_col, int ** block, int my_rank, int num_processes, MPI_Comm row_comm, MPI_Comm col_comm);
int test_results(int first_row, int num_rows, int first_col, int num_cols, int ** A);

/* Fast Integer square root */
unsigned long isqrt(unsigned long x){
    register unsigned long op, res, one;
    op = x;
    res = 0;

    /* "one" starts at the highest power of four <= than the argument.*/
    one = 1 << 30; /* second-to-top bit set */
    while (one > op) one >>= 2;

    while (one != 0) {
        if (op >= res + one) {
            op -= res + one;
            res += one << 1; // <-- faster than 2 * one
        }
        res >>= 1;
        one >>= 2;
    }
    return res;
}

/* Sequential Floyd all-pairs shortest path
algorithm used to check results on one block */
int ** floyd(int N, int ** A){
    for (int k=0; k<N; k++) {
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                A[i][j] = min(A[i][j], A[i][k] + A[k][j]);
            }
        }
    }
    return A;
}

/* Test results - return 0 if ok, return 1 if error */
int test_results(int first_row, int num_rows, int first_col, int num_cols, int ** A) {
    int blockSize = num_rows * num_cols;
    int n = SIZE;

    /*
    * B is a 1D array that contains the entire N x N matrix - initialize B
    * and run sequential floyd's algorithm on B to check results
    */

    int ** B = (int **) malloc(n * sizeof(int*));
    B[0] = (int *) malloc((n * n) * sizeof(int));
    for( int i = 1; i < n; i++ ) {
        B[i] = B[i-1] + n;
    }

    initMatrix(0, n, 0, n, B);
    floyd(n, B);
    int error = 0;
    for (int i = 0, ii = first_row; i < num_rows; i++, ii++) {
        for (int j = 0, jj = first_col; j < num_cols; j++, jj++) {
            // Uncomment the prints to view results for small test cases
            // printf("got: %d, expected: %d, blockSize = %d\n", A[i][j],
            B[ii][jj], blockSize);
            // fflush(stdout);

            if ( A[i][j] != B[ii][jj] ) {
                error = 1;
                free(B[0]);
                free(B);
                return error;
            }
        }
    }

    free(B[0]);
    free(B);
    return error;
}

/* Initialize Connectivity Matrix */
void initMatrix(int first_row, int num_rows, int first_col, int num_cols, int ** A) {
    for (int i = 0, ii = first_row; i < num_rows; i++, ii++) {
        for (int j = 0, jj = first_col; j < num_cols; j++, jj++) {
            A[i][j] = 100;

            // TODO - Change this later to "infinity"
            if ( ((jj+1) % (ii+1)) == 0 )
                A[i][j] = 1;
            if ( ((ii+1) % (jj+1)) == 0 )
                A[i][j] = 1;
            if (ii == jj)
                A[i][j] = 0;
        }
    }
}

/* Floyd 2D source-partitioned parallel algorithm for all-pairs shortest path */
int ** floyd2D(int num_rows, int first_row, int first_col,
    int ** block, int my_rank, int num_processes,
    MPI_Comm row_comm, MPI_Comm col_comm)
{
    MPI_Status status;
    double comm_time, total_comm_time =0.0;
    int nodes_per_row = isqrt(num_processes);
    int nodes_per_col = nodes_per_row;
    int num_cols = num_rows;
    int row_rank = my_rank % nodes_per_row;
    int col_rank = my_rank / nodes_per_row;
    int root_row = 0; // rank of the node that has part of the kth row
    int root_col = 0; // rank of the node that has part of the kth col
    int * col = (int *) malloc(num_rows * sizeof(int));
    int * row = (int *) malloc(num_cols * sizeof(int));

    //Get thread_id
    int thread_id = omp_get_thread_num();

    for (int k = 0; k < SIZE; k++) {
        if (my_rank == 0 && thread_id == 0)
            comm_time = MPI_Wtime();

        /* --------------------- Start of communication---------------------------*/
        if (k && ((k % num_rows) == 0)) {
            root_row++;
            root_col++;
        }

        // If this node has a portion of kth row, copy to row buf and send
        if (col_rank == root_col) {
            // Copy to row buf
            int row_index = k % num_rows;
            for (int j = 0; j < num_cols; j++)
                row[j] = block[row_index][j];

            // Process (i,j) sends their part of kth row to (i-1,j) and (i+1,j)
            if (col_rank != 0)
                MPI_Send(row, num_rows, MPI_INT, col_rank - 1, 1, col_comm);

            if (col_rank != nodes_per_row - 1)
                MPI_Send(row, num_rows, MPI_INT, col_rank + 1, 1, col_comm);
        }
        // If node doesn't have portion of kth row, receive & forward
        else {
            if (col_rank > root_col) {
                MPI_Recv(row, num_rows, MPI_INT, col_rank - 1, 1, col_comm, &status);

                if (col_rank != nodes_per_row - 1)
                    MPI_Send(row, num_rows, MPI_INT, col_rank + 1, 1, col_comm);
            }
            else if (col_rank < root_col) {
                MPI_Recv(row, num_rows, MPI_INT, col_rank + 1, 1, col_comm, &status);
                if (col_rank != 0)
                    MPI_Send(row, num_rows, MPI_INT, col_rank - 1, 1, col_comm);
            }
        }

        // If this node has a portion of kth col, copy to col buf and send
        if (row_rank == root_row) {
            // Copy to col buf
            int col_index = k % num_cols;
            for (int j = 0; j < num_cols; j++)
                col[j] = block[j][col_index];

            // Process (i,j) sends their part of kth col to (i,j+1) and (i,j-1)
            if (row_rank != 0)
                MPI_Send(col, num_cols, MPI_INT, row_rank - 1, 1, row_comm);
            if (row_rank != nodes_per_row - 1)
                MPI_Send(col, num_cols, MPI_INT, row_rank + 1, 1, row_comm);
        }
        // If node doesn't have portion of kth column, receive & forward
        else {
            if (row_rank > root_row) {
                MPI_Recv(col, num_cols, MPI_INT, row_rank - 1, 1, row_comm, &status);
                if (row_rank != nodes_per_row - 1)
                    MPI_Send(col, num_cols, MPI_INT, row_rank + 1, 1, row_comm);
            }
            else if (row_rank < root_row) {
                MPI_Recv(col, num_cols, MPI_INT, row_rank + 1, 1, row_comm, &status);
                if (row_rank != 0)
                MPI_Send(col, num_cols, MPI_INT, row_rank - 1, 1, row_comm);
            }
        }

        /* --------------------- End of communication---------------------------*/
        //#pragma omp barrier
        if (my_rank == 0 && thread_id == 0) {
            comm_time = MPI_Wtime() - comm_time;
            total_comm_time += comm_time;

            // Perform algorithm on block
            #pragma omp for schedule(guided)
            for (int i = 0; i < num_rows; i++) {
                for (int j = 0; j < num_cols; j++) {
                    block[i][j] = min(block[i][j], col[i] + row[j]);
                }
            }
        }

        if (my_rank == 0 && thread_id == 0) {
            printf("Comm time = %1.6f\n",total_comm_time);
            fflush(stdout);
        }
        return block;
    }
}
