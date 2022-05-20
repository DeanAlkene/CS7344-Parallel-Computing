#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int My_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
    int id;
    int p;
    int i;
    int ret;
    int type_size;
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    MPI_Status* status;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    send_requests = (MPI_Request *)malloc(p * sizeof(MPI_Request));
    recv_requests = (MPI_Request *)malloc(p * sizeof(MPI_Request));
    status = (MPI_Status *)malloc(p * sizeof(MPI_Status));

    for (i = 0; i < p; ++i) {
        if (i != id) {
            ret = MPI_Isend(sendbuf, sendcount, sendtype, i, 0, MPI_COMM_WORLD, &send_requests[i]);
            ret = MPI_Irecv(recvbuf + (i * recvcount), recvcount, recvtype, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
        } else {
            MPI_Type_size(recvtype, &type_size);
            memcpy(recvbuf + (i * recvcount), sendbuf, recvcount * type_size);
        }
    }

    for (i = 0; i < p; ++i) {
        if (i != id) {
            ret = MPI_Wait(&send_requests[i], &status[i]);
            ret = MPI_Wait(&recv_requests[i], &status[i]);
        }
    }

    free(send_requests);
    free(recv_requests);
    free(status);
    return ret;
}

int main(int argc, char *argv[]) {
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    int n;                 /* Elements per process (identical) */
    char* in_buf;
    char* out_buf;
    int i;
    int mode;
    double elapsed_time;

    MPI_Init(&argc, &argv);

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
        if (!id) printf("mode = 0: Allgather, mode = 1: my Allgather\n");
        MPI_Finalize();
        exit(1);
    }
    out_buf = (char *) malloc(n);
    if (out_buf == NULL) {
        if (!id) printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < n; ++i) {
        out_buf[i] = 'a' + id;
    }

    in_buf = (char *) malloc(p * n);
    if (in_buf == NULL) {
        if (!id) printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();
    if (!mode) {
        MPI_Allgather(out_buf, n, MPI_CHAR, in_buf, n, MPI_CHAR, MPI_COMM_WORLD);
    } else {
        My_Allgather(out_buf, n, MPI_CHAR, in_buf, n, MPI_CHAR, MPI_COMM_WORLD);
    }
    elapsed_time += MPI_Wtime();

    if (n * p <= 160) {
        printf("Process %d: ", id);
        for (i = 0; i < p * n; ++i) {
            printf("%c", in_buf[i]);
        }
        printf("\n");
    }

    if (!id) {
        printf("Total elapsed time: %10.6f\n", elapsed_time);
        fflush(stdout);
    }

    free(in_buf);
    free(out_buf);
    MPI_Finalize();
    return 0;
}