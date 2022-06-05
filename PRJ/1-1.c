#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int My_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
    int id;                      /* Process rank */
    int p;                       /* Number of processes */
    int i;                       /* Loop index */
    int ret;                     /* Return value */
    int type_size;               /* Size of recv type */
    MPI_Request* send_requests;  /* Array of send requests */
    MPI_Request* recv_requests;  /* Array of recv requests */
    MPI_Status* status;          /* send/recv status */
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    send_requests = (MPI_Request *)malloc(p * sizeof(MPI_Request));
    recv_requests = (MPI_Request *)malloc(p * sizeof(MPI_Request));
    status = (MPI_Status *)malloc(p * sizeof(MPI_Status));

    for (i = 0; i < p; ++i) {
        if (i != id) {
            // send buf to all other procs
            ret = MPI_Isend(sendbuf, sendcount, sendtype, i, 0, MPI_COMM_WORLD, &send_requests[i]);
            // recv buf from all other procs and put in the right place
            ret = MPI_Irecv(recvbuf + (i * recvcount), recvcount, recvtype, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
        } else {
            // local: copy from send buf to the right place in the recv buf
            MPI_Type_size(recvtype, &type_size);
            memcpy(recvbuf + (i * recvcount), sendbuf, recvcount * type_size);
        }
    }

    // wait for all async send&recv
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
    char* in_buf;          /* Receive buffer */
    char* out_buf;         /* Send buffer */
    int i;                 /* Loop index */
    int mode;              /* 0 - MPI_Allgather, 1 - My_Allgather*/
    double elapsed_time;   /* Elapsed time */

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
    // fill with some characters
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
        printf("Elapsed time: %10.3f ms\n", elapsed_time * 1000);
        fflush(stdout);
    }

    free(in_buf);
    free(out_buf);
    MPI_Finalize();
    return 0;
}