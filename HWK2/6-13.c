#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT
#define LIVE_CELL 1
#define DEAD_CELL 0

int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

int check_valid(int x, int y, int m, int n) {
    return (x >= 0 && x < m && y >= 0 && y < n);
}

int count_in_single_row(int j, int n, dtype* row) {
    int cur_j, live_cnt;
    live_cnt = 0;
    for (cur_j = j - 1; cur_j <= j + 1; ++cur_j) {
        if (cur_j >= 0 && cur_j < n && row[cur_j] == LIVE_CELL) {
            live_cnt++;
        }
    }
    return live_cnt;
}

int count_in_map(int i, int j, int local_m, int n, dtype **map, int dir_i, int dir_j) {
    int cur_i, cur_j, k, live_cnt;
    live_cnt = 0;
    for (k = dir_i; k <= dir_j; ++k) {
        cur_i = i + dir[k][0];
        cur_j = j + dir[k][1];
        if (check_valid(cur_i, cur_j, local_m, n) && map[cur_i][cur_j] == LIVE_CELL) {
            live_cnt++;
        }
    }
    return live_cnt;
}

void update_state(int id, int p, dtype **cur_map, dtype **next_map, int m, int n) {
    int i, j, k;
    int cur_i, cur_j;
    int local_m;           /* Local rows */
    int first_row;         /* Index of the first row */
    int last_row;          /* Index of the last row */
    int live_cnt;          /* Count for live neighbors */
    dtype* up_row;         /* Hold the received row from id - 1 */
    dtype* down_row;       /* Hold the received row from id + 1 */
    MPI_Status status;

    up_row = (dtype *) my_malloc(id, n * sizeof(dtype));
    down_row = (dtype *) my_malloc(id, n * sizeof(dtype));
    // set default as dead cells (for id = 0 and id = p - 1)
    memset(up_row, DEAD_CELL, n * sizeof(dtype));
    memset(down_row, DEAD_CELL, n * sizeof(dtype));

    local_m = BLOCK_SIZE(id, p, m);
    first_row = 0;
    last_row = local_m - 1;

    // send last_row to id + 1
    if (id != p - 1) {
        MPI_Send((void *) cur_map[last_row], n, MPI_TYPE, id + 1, 0, MPI_COMM_WORLD);
    }
    // receive last_row from id - 1
    if (id != 0) {
        MPI_Recv(up_row, n, MPI_TYPE, id - 1, 0, MPI_COMM_WORLD, &status);
    }
    // send first_row to id - 1
    if (id != 0) {
        MPI_Send((void *) cur_map[first_row], n, MPI_TYPE, id - 1, 0, MPI_COMM_WORLD);
    }
    // receive first_row from id + 1
    if (id != p - 1) {
        MPI_Recv(down_row, n, MPI_TYPE, id + 1, 0, MPI_COMM_WORLD, &status);
    }

    for (i = 0; i < local_m; i++) {
        for (j = 0; j < n; j++) {
            // count for live cells
            live_cnt = 0;
            if (i == 0) { // first row locally
                // count in received 'last_row' from id - 1
                live_cnt += count_in_single_row(j, n, up_row);
                // if there're more than 1 rows locally, directly count in cur_map
                if (local_m > 1) {
                    live_cnt += count_in_map(i, j, local_m, n, cur_map, 3, 7);
                } else {
                    // only one row locally, count in row 0 firstly
                    live_cnt += count_in_map(i, j, local_m, n, cur_map, 3, 4);
                    // ... then count in received 'first_row' of id + 1
                    live_cnt += count_in_single_row(j, n, down_row);
                }
            } else if (i == local_m - 1) { // last row locally
                // if there're more than 1 rows locally, directly count in cur_map
                if (local_m > 1) {
                    live_cnt += count_in_map(i, j, local_m, n, cur_map, 0, 4);
                } else { // this branch NEVER executes, just for symmetric
                    // only one row locally, count in received 'last_row' of id - 1 firstly
                    live_cnt += count_in_single_row(j, n, up_row);
                    // ... then count in count in row local_m - 1
                    live_cnt += count_in_map(i, j, local_m, n, cur_map, 3, 4);
                }
                // count in received 'last_row' from id + 1
                live_cnt += count_in_single_row(j, n, down_row);
            } else {
                live_cnt += count_in_map(i, j, local_m, n, cur_map, 0, 7);
            }
            // update cell state
            if (cur_map[i][j] == DEAD_CELL) {
                if (live_cnt == 3) {
                    next_map[i][j] = LIVE_CELL;
                } else {
                    next_map[i][j] = DEAD_CELL;
                }
            } else {
                if (live_cnt == 2 || live_cnt == 3) {
                    next_map[i][j] = LIVE_CELL;
                } else {
                    next_map[i][j] = DEAD_CELL;
                }
            }
        }
    }
    free(up_row);
    free(down_row);
}

int main(int argc, char *argv[]) {
    dtype** a;             /* Doubly subscripted array */
    dtype* storage_a;      /* Local portion of array elements */
    dtype** b;             /* Doubly subscripted array */
    dtype* storage_b;      /* Local portion of array elements */
    void** lptr;           /* Pointer into 'b' */
    void* rptr;            /* Pointer into 'storage_b' */
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    int m;                 /* Rows in matrix */
    int n;                 /* Columns in matrix */
    int local_rows;        /* Local rows */
    int datum_size;        /* Size of matrix element */
    int i, j, k;
    dtype** cur_map;
    dtype** next_map;
    double elapsed_time;

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 4) {
        if (!id) printf("Command line: %s <path> <j> <k>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    j = atoi(argv[2]);
    k = atoi(argv[3]);

    read_row_striped_matrix(argv[1], (void *) &a, (void *) &storage_a, MPI_TYPE, &m, &n, MPI_COMM_WORLD);
    // if (!id) printf("Initial state: \n");
    // print_row_striped_matrix((void **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);

    local_rows = BLOCK_SIZE(id, p, m);
    datum_size = get_size(MPI_TYPE);

    // allocate for another memory block for update
    storage_b = (dtype *) my_malloc (id, local_rows * n * datum_size);
    b = (dtype **) my_malloc (id, local_rows * PTR_SIZE);

    lptr = (void *) &(b[0]);
    rptr = (void *) storage_b;
    for (i = 0; i < local_rows; i++) {
        *(lptr++)= (void *) rptr;
        rptr += n * datum_size;
    }

    for (i = 0; i < j; i++) {
        // exchange the working map and the map to update to
        if (!(i % 2)) {
            cur_map = a;
            next_map = b;
        } else {
            cur_map = b;
            next_map = a;
        }
        update_state(id, p, cur_map, next_map, m, n);
        // if (!(i % k)) {
        //     if (!id) printf("Iter %d: \n", i);
        //     print_row_striped_matrix((void **) next_map, MPI_TYPE, m, n, MPI_COMM_WORLD);
        // }
    }

    elapsed_time += MPI_Wtime();

    if (!id) {
        printf("Total elapsed time: %10.6f\n", elapsed_time);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}