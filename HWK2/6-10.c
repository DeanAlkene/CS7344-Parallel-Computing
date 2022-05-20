#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int My_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int id;
    int p;
    int i;
    int ret;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (id == root) {
        for (i = 0; i < p; ++i) {
            if (i != root) {
                ret = MPI_Send(buffer, count, datatype, i, 0, comm);
            }
        }
    } else {
        ret = MPI_Recv(buffer, count, datatype, root, 0, comm, &status);
    }
    return ret;
}

int main(int argc, char *argv[]) {
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    int n;
    char* buf;
    int mode;
    double elapsed_time;

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 3) {
        if (!id) printf("Command line: %s <n> <mode>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    n = atoi(argv[1]);
    mode = atoi(argv[2]);
    if (mode != 0 && mode != 1) {
        if (!id) printf("mode = 0: Bcast, mode = 1: my Bcast\n");
        MPI_Finalize();
        exit(1);
    }
    buf = (char *) malloc(n);
    if (buf == NULL) {
        if (!id) printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    memset(buf, 0, n);

    if (!mode) {
        MPI_Bcast(buf, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        My_Bcast(buf, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    elapsed_time += MPI_Wtime();

    if (!id) {
        printf("Total elapsed time: %10.6f\n", elapsed_time);
        fflush(stdout);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}