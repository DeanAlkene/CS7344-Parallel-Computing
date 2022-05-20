#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "MyMPI.h"

typedef float dtype;
#define mpitype MPI_FLOAT
#define mpipairtype MPI_FLOAT_INT

struct pair {
    dtype cost;
    int root;
};

void alloc_matrix(void ***a, void **storage, int m, int n, int size) {
    int i;
    *storage = (void *) malloc(m * n * size);
    memset(*storage, 0, m * n * size);
    *a = (void **) malloc(m * sizeof(void *));
    for (i = 0; i < m; i++) {
        (*a)[i] = *storage + i * n * size;
    }
}

void print_root(int **root, int low, int high) {
    printf("Root of tree spanning %d-%d is %d\n", low, high, root[low][high + 1]);
    if (low < root[low][high + 1] - 1) 
        print_root(root, low, root[low][high + 1] - 1);
    if (root[low][high + 1] < high - 1) 
        print_root(root, root[low][high + 1] + 1, high);
}

int main(int argc, char *argv[]) {
    dtype *prob;                    /* Probability list */
    dtype **cost;                   /* The cost array */
    int **root;                     /* The root array */
    int id;                         /* Process ID number */
    int i, j, low, high;            /* Loop indices */
    int r;                          /* Root indices */
    int local_els;                  /* Cols of 'root' and 'cost' held by this process */
    int local_low;                  /* Start col index of 'root' and 'cost' held by this process */
    int local_high;                 /* End col index of 'root' and 'cost' held by this process */
    struct pair local_best_pair;    /* Local min cost and corresponding root */
    struct pair global_best_pair;   /* Global min cost and corresponding root */
    dtype rcost;                    /* Cost when root is r */
    double elapsed_time;            /* Parallel execution time */

    int n;                          /* Number of keys */
    int p;                          /* Number of processes */
    dtype *storage_cost;            /* This process's portion of 'cost' */
    int *storage_root;              /* This process's portion of 'root' */

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);   
    elapsed_time = 0.0;
    elapsed_time = -MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    if (argc != 2) {
        if (!id) printf("Command line: %s <prob>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    read_replicated_vector(argv[1], (void **) &prob, mpitype, &n, MPI_COMM_WORLD);
    // print_replicated_vector((void *) prob, mpitype, n, MPI_COMM_WORLD);

    alloc_matrix((void ***) &cost, (void **) &storage_cost, n + 1, n + 1, sizeof(dtype));
    alloc_matrix((void ***) &root, (void **) &storage_root, n + 1, n + 1, sizeof(int));

    for (low = n; low >= 0; low--) {
        cost[low][low] = 0.0;
        root[low][low] = low;
        for (high = low + 1; high <= n; high++) {
            // [local_low, local_high) + low
            local_low = BLOCK_LOW(id, p, high - low);
            local_high = BLOCK_HIGH(id, p, high - low) + 1;
            local_els = BLOCK_SIZE(id, p, high - low);
            local_best_pair.cost = FLT_MAX;
            for (r = low + local_low; r < low + local_high; r++) {
                rcost = cost[low][r] + cost[r + 1][high];
                for (i = low; i < high; i++) rcost += prob[i];
                if (rcost < local_best_pair.cost) {
                    local_best_pair.cost = rcost;
                    local_best_pair.root = r;
                }
            }
            MPI_Allreduce(&local_best_pair, &global_best_pair, 1, mpipairtype, MPI_MINLOC, MPI_COMM_WORLD);
            cost[low][high] = global_best_pair.cost;
            root[low][high] = global_best_pair.root;
        }
    }

    elapsed_time += MPI_Wtime();

    if(!id) {
        // print_root(root, 0, n - 1);
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }

    free(prob);
    free(cost);
    free(root);
    free(storage_cost);
    free(storage_root);
    MPI_Finalize();
    return 0;
}