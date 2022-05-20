#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "MyMPI.h"

typedef double dtype;
#define mpitype MPI_DOUBLE

int main(int argc, char *argv[]) {
    dtype **a;             /* The matrix */
    dtype *b;              /* The vector */
    dtype *c;              /* The product vector */
    dtype *c_part;         /* Partial sum */
    int i, j;              /* Loop indices */
    int id;                /* Process ID number */
    int local_els;         /* Cols of 'a' held by this process */
    int local_low;         /* Start col index of 'a' held by this process */
    int local_high;        /* End col index of 'a' held by this process */
    double elapsed_time;   /* Parallel execution time */

    int m;                 /* Rows of 'a' */
    int n;                 /* Cols of 'a' */
    int nprime;            /* Size of 'b' */
    int p;                 /* Number of processes */
    dtype *storage;        /* This process's portion of 'a' */

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 3) {
        if (!id) printf("Command line: %s <matrix> <vector>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    
    // The matrix is distributed, read and print
    read_col_striped_matrix(argv[1], (void ***) &a, (void **) &storage, mpitype, &m, &n, MPI_COMM_WORLD);
    // print_col_striped_matrix((void **) a, mpitype, m, n, MPI_COMM_WORLD);
    // The vector is replicated, read and print
    read_replicated_vector(argv[2], (void **) &b, mpitype, &nprime, MPI_COMM_WORLD);
    // print_replicated_vector((void *) b, mpitype, nprime, MPI_COMM_WORLD);
    
    c_part = (dtype *) my_malloc(id, m * sizeof(dtype));
    local_els = BLOCK_SIZE(id, p, n);
    local_low = BLOCK_LOW(id, p, n);

    // Calculate partial result vector
    for (i = 0; i < m; i++) {
        c_part[i] = 0.0;
        for (j = 0; j < local_els; j++) {
            c_part[i] += a[i][j] * b[j + local_low];
        }
    }

    // Use allreduce to sum up all partial results
    c = (dtype *) my_malloc(id, m * sizeof(dtype));
    MPI_Allreduce(c_part, c, m, mpitype, MPI_SUM, MPI_COMM_WORLD);
    print_replicated_vector((void *) c, mpitype, m, MPI_COMM_WORLD);
    
    elapsed_time += MPI_Wtime();
    if (!id) {
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }

    free(a);
    free(storage);
    free(b);
    free(c);
    free(c_part);
    MPI_Finalize();
    return 0;
}
