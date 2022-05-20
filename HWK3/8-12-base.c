#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>

void print_root(int **root, int low, int high) {
    printf("Root of tree spanning %d-%d is %d\n", low, high, root[low][high + 1]);
    if (low < root[low][high + 1] - 1)
        print_root(root, low, root[low][high + 1] - 1);
    if (root[low][high + 1] < high - 1)
        print_root(root, root[low][high + 1] + 1, high);
}

void alloc_matrix(void ***a, int m, int n, int size) {
    int i;
    void *storage;
    storage = (void *) malloc(m * n * size);
    *a = (void **) malloc(m * sizeof(void *));
    for (i = 0; i < m; i++) {
        (*a)[i] = storage + i * n * size;
    }
}

int main (int argc, char *argv[]) {
    float bestcost;
    int bestroot;
    int high;
    int i, j;
    int low;
    int n;
    int r;
    float rcost;
    int **root;
    float **cost;
    float *p;

    scanf("%d", &n);
    p = (float *) malloc(n * sizeof(float));
    for (i = 0; i < n; i++) {
        scanf("%f", &p[i]);
    }

    alloc_matrix((void ***) &cost, n + 1, n + 1, sizeof(float));
    alloc_matrix((void ***) &root, n + 1, n + 1, sizeof(int));
    for (low = n; low >= 0; low--) {
        cost[low][low] = 0.0;
        root[low][low] = low;
        for (high = low + 1; high <= n; high++) {
            bestcost = FLT_MAX;
            for (r = low; r < high; r++) {
                rcost = cost[low][r] + cost[r + 1][high];
                for (j = low; j < high; j++) rcost += p[j];
                if (rcost < bestcost) {
                    bestcost = rcost;
                    bestroot = r;
                }
            }
            cost[low][high] = bestcost;
            root[low][high] = bestroot;
        }
    }
    for (i = 0; i < n + 1; i++) {
        for (j = 0; j < n + 1; j++) {
            printf("%d ", root[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < n + 1; i++) {
        for (j = 0; j < n + 1; j++) {
            printf("%0.2f ", cost[i][j]);
        }
        printf("\n");
    }
    print_root(root, 0, n - 1);
    free(root);
    free(cost);
    free(p);
    return 0;
}