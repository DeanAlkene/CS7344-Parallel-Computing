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
#define REAL_LOW(len, n) ((n) - (len) - 1)

void alloc_matrix(void ***a, void **storage, int m, int n, int size) {
    int i;
    *storage = (void *) malloc(m * n * size);
    memset(*storage, 0, m * n * size);
    *a = (void **) malloc(m * sizeof(void *));
    for (i = 0; i < m; i++) {
        (*a)[i] = *storage + i * n * size;
    }
}

void print_root(int **root, int low, int high, int n) {
    printf("Root of tree spanning %d-%d is %d\n", low, high, root[REAL_LOW(high - low + 1, n)][high + 1]);
    if (low < root[REAL_LOW(high - low + 1, n)][high + 1] - 1)
        print_root(root, low, root[REAL_LOW(high - low + 1, n)][high + 1] - 1, n);
    if (root[REAL_LOW(high - low + 1, n)][high + 1] < high - 1)
        print_root(root, root[REAL_LOW(high - low + 1, n)][high + 1] + 1, high, n);
}

int main(int argc, char *argv[]) {
    dtype *prob;                    /* Probability list */
    dtype **cost;                   /* The cost array */
    int **root;                     /* The root array */
    int **global_root;              /* The global root array on p - 1 */
    int id;                         /* Process ID number */
    int i, j, len, low, high;       /* Loop indices */
    int r;                          /* Root indices */
    int local_els;                  /* Cols of 'root' and 'cost' held by this process */
    int local_low;                  /* Start col index of 'root' and 'cost' held by this process */
    int local_high;                 /* End col index of 'root' and 'cost' held by this process */
    dtype bestcost;                 /* Min cost */
    int bestroot;                   /* Corresponding root of the min cost */
    dtype rcost;                    /* Cost when root is r */
    int *count;                     /* For Allgather */
    int *disp;                      /* For Allgather */
    int tmp_low, tmp_high;          /* For Allgather */
    double elapsed_time;            /* Parallel execution time */

    int n;                          /* Number of keys */
    int p;                          /* Number of processes */
    dtype *storage_cost;            /* This process's portion of 'cost' */
    int *storage_root;              /* This process's portion of 'root' */
    int *global_storage_root;       /* Global 'root' on p - 1 */

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (!id) printf("Command line: %s <prob>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    read_replicated_vector(argv[1], (void **) &prob, mpitype, &n, MPI_COMM_WORLD);
    // print_replicated_vector((void *) prob, mpitype, n, MPI_COMM_WORLD);

    local_low = BLOCK_LOW(id, p, n + 1);
    local_high = BLOCK_HIGH(id, p, n + 1);
    local_els = BLOCK_SIZE(id, p, n + 1);
    
    alloc_matrix((void ***) &cost, (void **) &storage_cost, n + 1, n + 1, sizeof(dtype));
    alloc_matrix((void ***) &root, (void **) &storage_root, n + 1, local_els, sizeof(int));
    if (id == p - 1) {
        alloc_matrix((void ***) &global_root, (void **) &global_storage_root, n + 1, n + 1, sizeof(int));
    } else {
        global_root = root;
        global_storage_root = storage_root;
    }
    count = my_malloc(id, p * sizeof(int));
    disp = my_malloc(id, p * sizeof(int));

    for (len = 0; len <= n; len++) {
        if (len <= local_high) {
            for (high = local_high; high >= len && high >= local_low; high--) {
                if (len == 0) {
                    cost[REAL_LOW(len, n + 1)][high] = 0.0;
                    root[REAL_LOW(len, n + 1)][high - local_low] = high;
                    continue;
                }
                low = high - len;
                bestcost = FLT_MAX;
                for (r = low; r < high; ++r) {
                    rcost = cost[REAL_LOW(r - low, n + 1)][r] + cost[REAL_LOW(high - r - 1, n + 1)][high];
                    for (i = low; i < high; i++) rcost += prob[i];
                    if (rcost < bestcost) {
                        bestcost = rcost;
                        bestroot = r;
                    }
                }
                cost[REAL_LOW(len, n + 1)][high] = bestcost;
                root[REAL_LOW(len, n + 1)][high - local_low] = bestroot;
            }
        }
        // send and receive
        tmp_low = BLOCK_LOW(0, p, n + 1);
        tmp_high = BLOCK_HIGH(0, p, n + 1);
        count[0] = MAX(tmp_high - MAX(len, tmp_low) + 1, 0);
        disp[0] = len;
        for (i = 1; i < p; i++) {
            disp[i] = disp[i - 1] + count[i - 1];
            tmp_low = BLOCK_LOW(i, p, n + 1);
            tmp_high = BLOCK_HIGH(i, p, n + 1);
            count[i] = MAX(tmp_high - MAX(len, tmp_low) + 1, 0);
        }
        
        MPI_Allgatherv(cost[REAL_LOW(len, n + 1)] + MAX(len, local_low), count[id], mpitype, cost[REAL_LOW(len, n + 1)], count, disp, mpitype, MPI_COMM_WORLD);
        MPI_Gatherv(root[REAL_LOW(len, n + 1)] + MAX(len - local_low, 0), count[id], MPI_INT, global_root[REAL_LOW(len, n + 1)], count, disp, MPI_INT, p - 1, MPI_COMM_WORLD);
    }

    elapsed_time += MPI_Wtime();

    if (id == p - 1) {
        // print_root(global_root, 0, n - 1, n + 1);
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }
    
    free(prob);
    free(cost);
    free(root);
    free(count);
    free(disp);
    free(storage_cost);
    free(storage_root);
    if (id == p - 1) {
        free(global_root);
        free(global_storage_root);
    }
    MPI_Finalize();
    return 0;
}
