#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "MyMPI.h"

#define ROW_MSG 0
#define VECTOR_MSG 1
#define SIZE_MSG 3
#define RES_MSG 4

#define MATRIX_ARG 1
#define VECTOR_ARG 2

typedef double dtype;
#define mpitype MPI_DOUBLE

void read_vector (
    int          id,     /* IN - Process ID */
    char        *s,      /* IN - File name */
    void       **v,      /* OUT - Vector */
    MPI_Datatype dtype,  /* IN - Vector type */
    int         *n)      /* OUT - Vector length */
{
    int        datum_size; /* Bytes per vector element */
    int        i;
    FILE      *infileptr;  /* Input file pointer */

    datum_size = get_size (dtype);
    infileptr = fopen (s, "r");
    if (infileptr == NULL) *n = 0;
    else fread (n, sizeof(int), 1, infileptr);
    if (! *n) MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);
    *v = my_malloc (id, *n * datum_size);
    fread (*v, datum_size, *n, infileptr);
    fclose (infileptr);
}

void read_matrix (
    int          id,       /* IN - Process ID */
    char        *s,        /* IN - File name */
    void      ***subs,     /* OUT - 2D matrix indices */
    void       **storage,  /* OUT - Matrix stored here */
    MPI_Datatype dtype,    /* IN - Matrix element type */
    int         *m,        /* OUT - Matrix rows */
    int         *n)        /* OUT - Matrix cols */
{
    int          datum_size;   /* Size of matrix element */
    int          i;
    FILE        *infileptr;    /* Input file pointer */
    int          local_rows;   /* Rows on this proc */
    void       **lptr;         /* Pointer into 'subs' */
    void        *rptr;         /* Pointer into 'storage' */
    int          x;            /* Result of read */

    datum_size = get_size (dtype);
    infileptr = fopen (s, "r");
    if (infileptr == NULL) *m = 0;
    else {
        fread (m, sizeof(int), 1, infileptr);
        fread (n, sizeof(int), 1, infileptr);
    }      
    if (!(*m)) MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);

    *storage = (void *) my_malloc (id, (*m) * (*n) * datum_size);
    *subs = (void **) my_malloc (id, (*m) * PTR_SIZE);

    lptr = (void *) &(*subs[0]);
    rptr = (void *) *storage;
    for (i = 0; i < *m; i++) {
        *(lptr++)= (void *) rptr;
        rptr += *n * datum_size;
    }
    x = fread (*storage, datum_size, (*m) * (*n), infileptr);
    fclose (infileptr);
}

void manager(int argc, char *argv[], int p) {
    dtype **a;             /* The matrix */
    dtype *b;              /* The vector */
    dtype *c;              /* The product vector */
    dtype *c_part;
    int i;
    int m;                 /* Rows of 'a' */
    int n;                 /* Cols of 'a' */
    int nprime;            /* Size of 'b' */
    int low;
    int nrow;
    int res_size;
    int num_proc;
    int terminated;
    int src;
    int tag;
    MPI_Request pending;   /* Handle for recv request */
    MPI_Status status;     /* Message status */
    dtype *storage;        /* This process's portion of 'a' */

    read_vector(0, argv[VECTOR_ARG], (void *) &b, mpitype, &nprime);
    for (i = 1; i < p; i++) {
        MPI_Isend(b, nprime, mpitype, i, VECTOR_MSG, MPI_COMM_WORLD, &pending);
    }
    read_matrix(0, argv[MATRIX_ARG], (void ***) &a, (void **) &storage, mpitype, &m, &n);
    if (n != nprime) {
        free(a);
        free(b);
        free(storage);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_SIZE);
    }
    num_proc = MIN(m, p - 1);
    for (i = 0; i < p - 1; i++) {
        if (i < num_proc) {
            low = BLOCK_LOW(i, num_proc, m);
            nrow = BLOCK_SIZE(i, num_proc, m);
            MPI_Isend(storage + low * n, nrow * n, mpitype, i + 1, ROW_MSG, MPI_COMM_WORLD, &pending);
        } else {
            MPI_Isend(NULL, 0, mpitype, i + 1, ROW_MSG, MPI_COMM_WORLD, &pending);
        }
    }

    c_part = (dtype *) my_malloc(0, (m / num_proc + 1) * sizeof(dtype));
    c = (dtype *) my_malloc(0, m * sizeof(dtype));
    terminated = 0;
    do {
        MPI_Probe(MPI_ANY_SOURCE, RES_MSG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, mpitype, &res_size);
        MPI_Recv(c_part, res_size, mpitype, status.MPI_SOURCE, RES_MSG, MPI_COMM_WORLD, &status);
        src = status.MPI_SOURCE;
        tag = status.MPI_TAG;
        if (tag == RES_MSG) {
            low = BLOCK_LOW(src - 1, num_proc, m);
            for (i = 0; i < res_size; i++) {
                c[low + i] = c_part[i];
            }
        }
        terminated++;
    } while (terminated < num_proc);

    print_subvector(c, mpitype, m);
    printf("\n\n");

    free(a);
    free(b);
    free(storage);
    free(c_part);
    free(c);
}

void worker(int argc, char *argv[], MPI_Comm worker_comm) {
    int worker_id;         /* Process ID number */
    dtype *a;             /* The row-striped matrix */
    dtype *b;              /* The vector */
    dtype *c;              /* The product vector */
    int i, j;
    int m;                 /* Rows of 'a' */
    int n;                 /* Cols of 'a' */
    int nprime;            /* Size of 'b' */
    MPI_Request pending;   /* Handle for recv request */
    MPI_Status status;     /* Message status */

    MPI_Comm_rank(MPI_COMM_WORLD, &worker_id);

    MPI_Probe(0, VECTOR_MSG, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, mpitype, &nprime);
    b = (dtype *) my_malloc(worker_id, nprime * sizeof(dtype));
    MPI_Recv(b, nprime, mpitype, 0, VECTOR_MSG, MPI_COMM_WORLD, &status);

    MPI_Probe(0, ROW_MSG, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, mpitype, &m);
    if (m == 0) {
        free(b);
        return;
    }
    a = (dtype *) my_malloc(worker_id, m * sizeof(dtype));
    MPI_Recv(a, m, mpitype, 0, ROW_MSG, MPI_COMM_WORLD, &status);

    m /= nprime;
    n = nprime;
    c = (dtype *) my_malloc(worker_id, m * sizeof(dtype));
    for (i = 0; i < m; i++) {
        c[i] = 0.0;
        for (j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }

    // MPI_Send(c, m, mpitype, 0, RES_MSG, MPI_COMM_WORLD);
    MPI_Isend(c, m, mpitype, 0, RES_MSG, MPI_COMM_WORLD, &pending);

    free(a);
    free(b);
    free(c);
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

    if (argc != 3) {
        if (!id) printf("Command line: %s <matrix> <vector>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    } else if (p < 2) {
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
