#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW((id), p, n))
#define BLOCK_OWNER(index, p, n) (((p) * ((index) + 1) - 1) / (n))

int main(int argc, char *argv[]) {
    double elapsed_time;   /* Parallel execution time */
    int first;             /* Index of first multiple */
    int global_count;      /* Global prime count */
    int high_value;        /* Highest value on this proc */
    int i;
    int id;                /* Process ID number */
    int index;             /* Index of current prime */
    int low_value;         /* Lowest value on this proc */
    char *marked;          /* Portion of 2,...,'n' */
    int n;                 /* Sieving from 2,...,'n' */
    int sqrt_n;            /* Square root of n */
    int p;                 /* Number of processes */
    int prime;             /* Current prime */
    int size;              /* Elements in 'marked' */
    int sqrt_size;         /* Elements in [2, sqrt(n)] */
    int role_cnt;          /* Counter for prime numbers in [2, sqrt(n)] */
    int flag;              /* Flag to control the exit of the while loop */

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time -= MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (!id) printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    
    n = atoi(argv[1]);
    sqrt_n = (int) sqrt((double) n);
    size = n - 1;
    sqrt_size = sqrt_n - 1;
    low_value = sqrt_n + 1;
    high_value = n;

    marked = (char *) malloc(size);
    if (marked == NULL) {
        if (!id) printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    memset(marked, 0, size);

    // mark prime numbers between [2, sqrt(n)] using the Sieve of Eratosthenes
    index = 0;
    prime = 2;
    do {
        first = prime * prime - 2;
        for (i = first; i < sqrt_size; i += prime) {
            marked[i] = 1;
        }
        while (marked[++index]);
        prime = index + 2;
    } while (prime * prime <= sqrt_n);

    index = 0;
    role_cnt = 0;
    while (index < sqrt_size) {
        // find prime number for id in this turn
        flag = 0;
        while (index < sqrt_size) {
            if (!marked[index]) {
                // found a prime number, check if it is id-th proc's turn
                if ((role_cnt++ % p) == id) {
                    prime = index + 2;
                    index++;
                    flag = 1;
                    break;
                }
            }
            index++;
        }
        // break if index is out of [2, sqrt(n)] and sqrt(n) is not a prime number
        if (index >= sqrt_size && !flag) {
            break;
        }
        /* determine the first index between [sqrt(n) + 1, n] for prime to sieve
           follow the method in Chapter 5, assume 0 corresponds to low_value = sqrt(n) + 1
           then add back sqrt(n) + 1 as we are operating on the global index
        */
        if (prime * prime > low_value)
            first = prime * prime - low_value;
        else {
            if (!(low_value % prime)) first = 0;
            else first = prime - (low_value % prime);
        }
        // the real first index
        first += sqrt_size;
        // mark by prime
        for (i = first; i < size; i += prime) {
            marked[i] = 1;
        }
    }
    
    // do OR-reduce, update in-place for id = 0
    if (!id) {
        MPI_Reduce(MPI_IN_PLACE, marked + sqrt_size, size - sqrt_size, MPI_CHAR, MPI_LOR, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(marked + sqrt_size, NULL, size - sqrt_size, MPI_CHAR, MPI_LOR, 0, MPI_COMM_WORLD);
    }

    elapsed_time += MPI_Wtime();

    if (!id) {
        global_count = 0;
        for (i = 0; i < size; i++)
            if (!marked[i]) global_count++;
    }

    if (!id) {
        printf("%d primes are less than or equal to %d\n", global_count, n);
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }

    free(marked);
    MPI_Finalize();
    return 0;
}
