#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "MyMPI.h"

#define REQUEST_MSG 0
#define REPLY_MSG 1

typedef unsigned long long dtype;
#define mpitype MPI_UNSIGNED_LONG_LONG

int cmp_func(const void * num1, const void * num2) {
   if (*(dtype *) num1 > *(dtype *) num2) return 1;
   else return -1;
}

dtype check_prime(dtype num) {
    if (num == 1) return 0;

    dtype i;
    dtype upper_bound;

    upper_bound = (dtype)(sqrt((double)num));

    for (i = 2; i <= upper_bound; ++i) {
        if (num % i == 0) {
            return 0;
        }
    }
    return 1;
}

void manager(int argc, char *argv[], int p) {
    int i, j, k;
    int n;
    int num_proc;
    int *send_buf;
    int send_size;
    int recv_size;
    int num_checked;
    int terminated;
    int src;
    int tag;
    dtype perfect_nums[32];
    MPI_Request pending;   /* Handle for recv request */
    MPI_Status status;     /* Message status */

    num_proc = MIN(32, p - 1);
    send_buf = (int *) my_malloc(0, (32 / num_proc + 1) * sizeof(int));
    for (i = 0; i < p - 1; i++) {
        send_size = 0;
        if (i < num_proc) {
            for (j = i + 1; j <= 32; j += num_proc) {
                send_buf[send_size++] = j;
            }
            MPI_Isend(send_buf, send_size, MPI_INT, i + 1, REQUEST_MSG, MPI_COMM_WORLD, &pending);
        } else {
            MPI_Isend(NULL, 0, MPI_INT, i + 1, REQUEST_MSG, MPI_COMM_WORLD, &pending);
        }
    }

    terminated = 0;
    num_checked = 0;
    memset(perfect_nums, 0xff, 32 * sizeof(dtype));
    do {
        MPI_Probe(MPI_ANY_SOURCE, REPLY_MSG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, mpitype, &recv_size);
        MPI_Recv(perfect_nums + num_checked, recv_size, mpitype, status.MPI_SOURCE, REPLY_MSG, MPI_COMM_WORLD, &status);
        num_checked += recv_size;
        terminated++;
    } while (terminated < num_proc && num_checked <= 8);

    qsort(perfect_nums, 32, sizeof(dtype), cmp_func);
    for (i = 0; i < 8; i++) {
        printf("%llu ", perfect_nums[i]);
    }
    printf("\n");
    free(send_buf);
}

void worker(int argc, char *argv[], MPI_Comm worker_comm) {
    int worker_id;         /* Process ID number */
    int size;              /* Numbers need to be checked */
    dtype *send_buf;
    int *recv_buf;
    int i, j;
    dtype num;               /* 2^n - 1 */
    dtype res;               /* If 2^n - 1 is prime */
    MPI_Request pending;   /* Handle for recv request */
    MPI_Status status;     /* Message status */

    MPI_Comm_rank(MPI_COMM_WORLD, &worker_id);

    MPI_Probe(0, REQUEST_MSG, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    if (size == 0) return;
    recv_buf = (int *) my_malloc(worker_id, size * sizeof(int));
    MPI_Recv(recv_buf, size, MPI_INT, 0, REQUEST_MSG, MPI_COMM_WORLD, &status);

    j = 0;
    send_buf = (dtype *) my_malloc(worker_id, size * sizeof(dtype));
    for (i = 0; i < size; i++) {
        num = ((dtype)1 << recv_buf[i]) - 1;
        res = check_prime(num);
        if (res == 1) send_buf[j++] = num * ((dtype)1 << (recv_buf[i] - 1));
    }
    MPI_Isend(send_buf, j, mpitype, 0, REPLY_MSG, MPI_COMM_WORLD, &pending);

    free(send_buf);
    free(recv_buf);
}

int main(int argc, char *argv[]) {
    double elapsed_time;   /* Parallel execution time */
    int id;                /* Process ID number */
    int p;                 /* Number of processes */
    MPI_Comm worker_comm;  /* Worker-only communicator */

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p < 2) {
        printf("Process needs at least 2 processes\n");
    } else {
        if (!id) {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, id, &worker_comm);
            manager(argc, argv, p);
        } else {
            MPI_Comm_split(MPI_COMM_WORLD, 0, id, &worker_comm);
            worker(argc, argv, worker_comm);
        }
    }
    
    elapsed_time += MPI_Wtime();
    if (!id) {
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
