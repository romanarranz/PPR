#include "hybrid.h"

int main( int argc, char **argv ){
    int n = SIZE;
    int my_rank;
    int num_processes;

    // <== INICIALIZAR MPI
    // ====================================>
    MPI_Init(&argc, &argv);
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    /* Create a new communicator - 2d grid of processes */
    int size[2];
    size[0] = size[1] = 0;
    MPI_Dims_create(num_processes, 2, size);

    int periodic[2];
    periodic[0] = periodic[1] = 1;
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, size, periodic, 0, &grid_comm);

    /* Get cartesian coordinates of this process in 2D Grid */
    int coords[2];
    MPI_Cart_coords(grid_comm, my_rank, 2, coords);
    int my_row_id = coords[0];
    int my_col_id = coords[1];

    /* Break 2D grid communicator into a communicator for each row and each process */
    MPI_Comm col_comm;
    MPI_Comm row_comm;
    MPI_Comm_split(grid_comm, my_row_id, my_col_id, &row_comm);
    MPI_Comm_split(grid_comm, my_col_id, my_row_id, &col_comm);

    /* Get rank of process in row and col communicators */
    int col_rank;
    int row_rank;
    MPI_Comm_rank( row_comm, &row_rank );
    MPI_Comm_rank( col_comm, &col_rank );

    // <== INICIALIZAR DATOS
    // ====================================>
    /* Calculate indices of data this process is responsible for */
    int nodes_per_row = isqrt(num_processes);
    int nodes_per_col = nodes_per_row;
    int rows_per_process = n / nodes_per_row;
    int cols_per_process = n / nodes_per_col;
    int first_row = my_row_id * rows_per_process;
    int first_col = my_col_id * cols_per_process;
    int last_row = first_row + rows_per_process;
    int last_col = first_col + cols_per_process;
    int num_rows = last_row - first_row;
    int num_cols = last_col - first_col;

    /* Allocate Contiguous 2D arrays mapping each processor's data to a 1D row array */
    int ** block = (int **) malloc(num_rows * sizeof(int*));

    block[0] = (int *) malloc(num_rows * num_cols * sizeof(int));
    for( int i = 1; i < num_rows; i++ )
        block[i] = block[i-1] + num_rows;

    /* Fill matrix with connectivity information */
    initMatrix(first_row, num_rows, first_col, num_cols, block);

    // <== RUN TEST
    // ====================================>

    /* Run Floyd's algorithm many times to eliminate noise */
    double Mflop_s, avg_time = 0.0, total_time =0.0;
    int n_iterations = 10;

    #pragma omp parallel shared(block){
    int thread_id = omp_get_thread_num();
    /* warm-up */
    floyd2D(num_rows, first_row, first_col, block, my_rank, num_processes, row_comm, col_comm);
    if (my_rank == 0 && thread_id == 0) {
        /* record start time */
        total_time = MPI_Wtime( );
    }
    for( int i = 0; i < n_iterations; i++ )
        floyd2D(num_rows, first_row, first_col, block, my_rank, num_processes, row_comm, col_comm);
        if (my_rank == 0 && thread_id == 0) {
            /* measure execution time */
            total_time = MPI_Wtime( ) - total_time;
            avg_time = total_time/n_iterations;

            /* compute Mflop/s rate */
            Mflop_s = 1e-6* n_iterations * n * n * n / total_time;

            /* Print results */
            printf ("nprocs:\t%d\t nthreads:\t%d\t avg_time:\t%1.6f\t
            Million ops /s:\t%g\n", num_processes,NTHREADS,avg_time, Mflop_s);
        }
    } //End of parallel region

    /* Deallocate memory */
    free( block[0] );
    free( block );
    MPI_Finalize();

    return 0;
}
