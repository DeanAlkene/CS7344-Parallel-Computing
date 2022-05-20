#include "mpi.h"
#include <stdio.h>
#include <math.h>

#define TOTAL_INTERVALS 10000000

int main(int argc, char *argv[]) {
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    double area;          /* Local area */
    double global_area;   /* Global area */
    double xi;            /* Midpoint of interval */
    double ysum;          /* Sum of rectangle heights */
    int i;
    int low;
    int high;
    double elapsed_time;

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    low = id * TOTAL_INTERVALS / p;
    high = (id + 1) * TOTAL_INTERVALS / p;

    area = 0.0;
    ysum = 0.0;
    for (i = low; i < high; ++i) {
       xi = (1.0 / TOTAL_INTERVALS) * (i + 0.5);
       ysum += 4.0 / (1.0 + xi * xi);
    }
    area = ysum * (1.0 / TOTAL_INTERVALS);

    MPI_Reduce(&area, &global_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (!id) {
        // printf("Pi= %13.11f\n", global_area);
        printf("%.6f\n", elapsed_time);
        fflush(stdout);
    }
    // printf("Process %d: [%d, %d) %d\n", id, low, high, solution);
    MPI_Finalize();
    return 0;
}
