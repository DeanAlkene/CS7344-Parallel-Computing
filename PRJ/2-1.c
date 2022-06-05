#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int tn;                 /* Number of threads */
    int i;                  /* Loop index */
    double x, y;            /* Sample point (x, y) */
    int count;              /* Count of points inside the 1/4 circle */
    int samples;            /* Total number of sample points */
    double pi;              /* Estimation of pi */
    unsigned int seed;      /* Random seed */
    double elapsed_time;    /* Elapsed time */

    if (argc != 3) {
        printf("Command line: %s <num_threads> <samples>\n", argv[0]);
        exit(1);
    }
    
    tn = atoi(argv[1]);
    samples = atoi(argv[2]);
    count = 0;
    omp_set_num_threads(tn);

    elapsed_time = 0.0;
    elapsed_time -= omp_get_wtime();
    #pragma omp parallel private(seed)
    {
        // seed = omp_get_thread_num() * (unsigned int)time(NULL);
        // fixed seed
        seed = omp_get_thread_num();
        #pragma omp for private(x, y) reduction(+:count)
        for (i = 0; i < samples; ++i) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            // check and count
            if (x*x + y*y <= 1) {
                count++;
            }
        }
    }
    pi = 4 * (double)count / samples;
    elapsed_time += omp_get_wtime();

    printf("Pi: %f\n", pi);
    printf("Elapsed time: %10.3f ms\n", elapsed_time * 1000);

    return 0;
}