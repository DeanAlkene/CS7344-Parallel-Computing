#include "mpi.h"
#include <stdio.h>
#include <math.h>

int check_prime(int num) {
    int i;
    int upper_bound;

    upper_bound = (int)(sqrt((double)num));

    for (i = 2; i <= upper_bound; ++i) {
        if (num % i == 0) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    int solution;          /* Local solutions */
    int global_solution;   /* Global solutions */
    int i;
    int low;
    int high;
    int cur;
    int next;
    double elapsed_time;

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    low = 3 + id * 999997 / p;
    high = 3 + (id + 1) * 999997 / p;
    if (low % 2 == 0) {
        low += 1;
    }

    solution = 0;
    cur = 0;
    for (i = low; i < high; i += 2) {
       next = check_prime(i);
       solution += cur && next;
       cur = next;
    }

    MPI_Reduce(&solution, &global_solution, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (!id) {
        // printf("Solution: %d\n", global_solution);
        printf("%.6f\n", elapsed_time);
        fflush(stdout);
    }
    // printf("Process %d: [%d, %d) %d\n", id, low, high, solution);
    MPI_Finalize();
    return 0;
}
