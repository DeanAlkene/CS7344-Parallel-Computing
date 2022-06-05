#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DAMPING_FACTOR 0.85

struct Node {
    double page_rank;
    int out_degree;
    int* edge_list;
};

void print_graph(struct Node* graph, int n) {
    int i, j;
    for (i = 0; i < n; ++i) {
        printf("Node %d -> ", i);
        for (j = 0; j < graph[i].out_degree; ++j) {
            printf("%d ", graph[i].edge_list[j]);
        }
        printf("\n");
    }
}

void write_page_rank(struct Node* graph, int n) {
    int i;
    FILE* f;
    f = fopen("pagerank.txt", "w");

    for (i = 0; i < n; ++i) {
        fprintf(f, "Node %d: %f\n", i, graph[i].page_rank);
    }

    fclose(f);
}

void construct_graph(struct Node** graph, int n) {
    int i, j, k, seed, out_degree, v, max_od;
    *graph = (struct Node*)malloc(n * sizeof(struct Node));
    max_od = n > 10 ? 10 : n;

    #pragma omp parallel private(seed)
    {
        seed = omp_get_thread_num(); // fix graph structure when num_thread is fixed
        // seed = omp_get_thread_num() * (unsigned int)time(NULL);
        #pragma omp for private(j, k, v, out_degree)
        for (i = 0; i < n; ++i) {
            (*graph)[i].page_rank = 1.0;
            out_degree = rand_r(&seed) % max_od + 1;
            (*graph)[i].out_degree = out_degree;
            (*graph)[i].edge_list = (int *)malloc(out_degree * sizeof(int));
            // generate edges without duplication
            for (j = 0; j < out_degree; ++j) {
                v = rand_r(&seed) % n;
                for (k = 0; k < j; ++k) {
                    if ((*graph)[i].edge_list[k] == v) {
                        v = rand_r(&seed) % n;
                        k = -1;
                    }
                }
                (*graph)[i].edge_list[j] = v;
            }
        }
    }
}

void do_page_rank_single(struct Node* graph, int n, int iterations) {
    int i, j, k;
    double pr;
    double* sum_arr;

    sum_arr = (double *)malloc(n * sizeof(double));
    for (i = 0; i < iterations; ++i) {
        memset(sum_arr, 0, n * sizeof(double));
        for (j = 0; j < n; ++j) {
            // accumulate PageRank of successors
            pr = graph[j].page_rank / graph[j].out_degree;
            for (k = 0; k < graph[j].out_degree; ++k) {
                sum_arr[graph[j].edge_list[k]] += pr;
            }
        }
        // update PageRank
        for (j = 0; j < n; ++j) {
            graph[j].page_rank = 1 - DAMPING_FACTOR + DAMPING_FACTOR * sum_arr[j];
        }
    }

    free(sum_arr);
}

void do_page_rank(struct Node* graph, int n, int iterations) {
    int i, j, k;
    double pr;
    double* sum_arr;

    sum_arr = (double *)malloc(n * sizeof(double));
    for (i = 0; i < iterations; ++i) {
        // memset(sum_arr, 0, n * sizeof(double));
        #pragma omp parallel
        {   
            #pragma omp for
            for (j = 0; j < n; ++j) {
                sum_arr[j] = 0.0;
            }
            // #pragma omp for private(k, pr)
            #pragma omp for reduction(+:sum_arr[:n]) private(k, pr)
            for (j = 0; j < n; ++j) {
                // accumulate PageRank of successors in parallel
                pr = graph[j].page_rank / graph[j].out_degree;
                for (k = 0; k < graph[j].out_degree; ++k) {
                    // #pragma omp atomic
                    sum_arr[graph[j].edge_list[k]] += pr;
                }
            }
            // update PageRank in parallel
            #pragma omp for
            for (j = 0; j < n; ++j) {
                graph[j].page_rank = 1 - DAMPING_FACTOR + DAMPING_FACTOR * sum_arr[j];
            }
        }
    }

    free(sum_arr);
}

void clear_graph(struct Node* graph, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        free(graph[i].edge_list);
    }
    free(graph);
}

int main(int argc, char *argv[]) {
    int tn;                 /* Number of threads */
    int n;                  /* Number of nodes */
    int iterations;         /* Total iterations */
    struct Node* graph;     /* The graph */
    double elapsed_time;    /* Elapsed time */

    if (argc != 4) {
        printf("Command line: %s <num_threads> <nodes> <iters>\n", argv[0]);
        exit(1);
    }

    tn = atoi(argv[1]);
    n = atoi(argv[2]);
    iterations = atoi(argv[3]);
    omp_set_num_threads(tn);
    // generate a graph randomly
    construct_graph(&graph, n);
    // print_graph(graph, n);
    elapsed_time = 0.0;
    elapsed_time -= omp_get_wtime();
    // do_page_rank_single(graph, n, iterations);
    do_page_rank(graph, n, iterations);
    elapsed_time += omp_get_wtime();
    printf("Elapsed time: %10.3f ms\n", elapsed_time * 1000);

    write_page_rank(graph, n);
    clear_graph(graph, n);
    return 0;
}