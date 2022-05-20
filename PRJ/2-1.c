#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int tn;
    int i;
    double x, y;
    int count;
    int samples;
    double pi;
    unsigned int seed;
    double elapsed_time;

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
        seed = omp_get_thread_num() * (unsigned int)time(NULL);
        #pragma omp for private(x, y) reduction(+:count)
        for (i = 0; i < samples; ++i) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if (x*x + y*y <= 1) {
                count++;
            }
        }
    }
    elapsed_time += omp_get_wtime();

    pi = 4 * (double)count / samples;
    printf("Pi: %f\n", pi);
    printf("Elapsed time: %f\n", elapsed_time);

    return 0;
}