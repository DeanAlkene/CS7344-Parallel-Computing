#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void quick_sort(int *arr, int len) {
    int l, r, pivot, tmp;
    if (len <= 1) return;
    
    l = 0;
    r = len - 1;
    pivot = arr[len / 2];
    while (l <= r) {
        while (arr[l] < pivot) ++l;
        while (arr[r] > pivot) --r;
        if (l <= r) {
            tmp = arr[l];
            arr[l] = arr[r];
            arr[r] = tmp;
            ++l;
            --r;
        }
    }

    #pragma omp task default(none) firstprivate(arr, r)
    {
        quick_sort(arr, r + 1);
    }
    #pragma omp task default(none) firstprivate(arr, l, len)
    {
        quick_sort(arr + l, len - l);
    }
}

int main(int argc, char *argv[]) {
    int tn;
    int i;
    int n;
    int* arr;
    double elapsed_time;

    if (argc != 3) {
        printf("Command line: %s <num_threads> <elem_num>\n", argv[0]);
        exit(1);
    }

    tn = atoi(argv[1]);
    n = atoi(argv[2]);
    omp_set_num_threads(tn);
    srand((unsigned int)time(NULL));
    arr = (int *) malloc(n * sizeof(int));
    if (arr == NULL) {
        printf("Cannot allocate enough memory\n");
        exit(1);
    }

    for (i = 0; i < n; ++i) {
        arr[i] = rand();
    }

    elapsed_time = 0.0;
    elapsed_time -= omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            quick_sort(arr, n);
        }
    }
    elapsed_time += omp_get_wtime();

    for (i = 1; i < n; ++i) {
        if (arr[i - 1] > arr[i]) {
            printf("Error!\n");
            exit(1);
        }
    }
    if (n < 50) {
        for (i = 0; i < n; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    printf("Elapsed time: %f\n", elapsed_time);

    free(arr);
    return 0;
}