#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef double dtype;
#define mpitype MPI_DOUBLE
// typedef int dtype;
// #define mpitype MPI_INT

#define ARR_IDX(i, j, n) ((i)*(n)+(j))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

void read_matrix(char* path, int* m, int* n, dtype** M) {
    FILE* f;

    f = fopen(path, "r");
    if (f == NULL) {
        printf("Cannot open file %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, -1);
    } else {
        fread(m, sizeof(int), 1, f);
        fread(n, sizeof(int), 1, f);
    }

    if (*m == 0 || *n == 0) {
        printf("Matrix is empty\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    *M = (dtype*)malloc((*m) * (*n) * sizeof(dtype));
    if ((*M) == NULL) {
        printf("Cannot allocate enough memory\n");
    }
    fread(*M, sizeof(dtype), (*m) * (*n), f);
    fclose(f);
}

void print_matrix(dtype* M, int m, int n) {
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; ++j) {
            if (mpitype == MPI_DOUBLE) {
                printf("%6.3f ", (double)M[ARR_IDX(i, j, n)]);
            } else {
                if (mpitype == MPI_FLOAT) {
                    printf("%6.3f ", (float)M[ARR_IDX(i, j, n)]);
                } else if (mpitype == MPI_INT) {
                    printf("%d\t", (int)M[ARR_IDX(i, j, n)]);
                }
            }
        }
        printf("\n");
    }
}

void write_matrix(dtype* M, int m, int n, char* path) {
    FILE* f;
    int i, j;

    f = fopen(path, "w");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; ++j) {
            if (mpitype == MPI_DOUBLE) {
                fprintf(f, "%6.3f ", (double)M[ARR_IDX(i, j, n)]);
            } else {
                if (mpitype == MPI_FLOAT) {
                    fprintf(f, "%6.3f ", (float)M[ARR_IDX(i, j, n)]);
                } else if (mpitype == MPI_INT) {
                    fprintf(f, "%d\t", (int)M[ARR_IDX(i, j, n)]);
                }
            }
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void write_matrix_bin(dtype* M, int m, int n, char* path) {
    FILE* f;
    int i, j;

    f = fopen(path, "w");
    fwrite(&m, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(M, sizeof(dtype), m * n, f);
    fclose(f);
}

void im2col_hw(
    int h_in, int w_in, 
    int k_h, int k_w, 
    int stride_h, int stride_w,
    dtype* feature, dtype* kernel,
    dtype** A, dtype** B,
    int* h_out, int* w_out,
    int* m_A, int* n_A, int* m_B, int* n_B) 
{
    int i, j, r;
    int row_idx;
    // feature map - kernel shape check
    if ((h_in - k_h) % stride_h != 0 || (w_in - k_w) % stride_w != 0) {
        printf("Feature map needs padding, quit\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    *h_out = (h_in - k_h) / stride_h + 1;
    *w_out = (w_in - k_w) / stride_w + 1;

    *A = (dtype*)malloc((*h_out) * (*w_out) * k_h * k_w * sizeof(dtype));
    // (i, j) is the upper-left point of the sliding window
    for (i = 0; i <= h_in - k_h; i += stride_h) {
        for (j = 0; j <= w_in - k_w; j += stride_w) {
            for (r = 0; r < k_h; ++r) {
                row_idx = i / stride_h * (*w_out) + j / stride_w;
                memcpy(*A + ARR_IDX(row_idx, r * k_w, k_h * k_w), feature + ARR_IDX(i + r, j, w_in), k_w * sizeof(dtype));
            }
        }
    }
    *m_A = (*h_out) * (*w_out);
    *n_A = k_h * k_w;

    // as N=1, C_in=1, C_out=1, B.shape=(k_h*k_w, 1), just point to kernel
    *B = kernel;
    *m_B = k_h * k_w;
    *n_B = 1;
}

void matrix_mul_transposed_inner(dtype* A, dtype* B_T, dtype** C, int m_A, int n_A, int m_B, int n_B) {
    int i, j, k;
    if (m_A == 0 || n_A == 0 || m_B == 0 || n_B == 0) {
        printf("A or B are empty\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    if (n_A != m_B) {
        printf("n_A != m_B\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    *C = (dtype*)malloc(m_A * n_B * sizeof(dtype));
    memset(*C, 0, m_A * n_B * sizeof(dtype));
    for (i = 0; i < m_A; ++i) {
        for (j = 0; j < n_B; ++j) {
            for (k = 0; k < n_A; ++k) {
                (*C)[ARR_IDX(i, j, n_B)] += A[ARR_IDX(i, k, n_A)] * B_T[ARR_IDX(j, k, m_B)];
            }
        }
    }
}

void transpose(dtype* A, dtype** A_T, int m, int n) {
    int i, j;

    *A_T = (dtype*)malloc(n * m * sizeof(dtype));
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            (*A_T)[ARR_IDX(j, i, m)] = A[ARR_IDX(i, j, n)];
        }
    }
}

void interleave_rows(dtype* rows, dtype* rows_interleaved, int m, int n, int p) {
    int i, j;
    dtype *rptr, *wptr;
    size_t cpy_size;

    for (i = 0; i < p; ++i) {
        cpy_size = BLOCK_SIZE(i, p, n);
        rptr = rows + m * BLOCK_LOW(i, p, n);
        wptr = rows_interleaved + BLOCK_LOW(i, p, n);
        for (j = 0; j < m; ++j) {
            memcpy(wptr + j * n, rptr + j * cpy_size, cpy_size * sizeof(dtype));
        }
    }
}

void matrix_mul(int id, int p, dtype* A, dtype* B, dtype** C, int m_A, int n_A, int m_B, int n_B, MPI_Comm comm) {
    int grid_id;            /* Process rank */
    int grid_dim[2];        /* Dimensions of grid */
    int grid_coord[2];      /* Process coords */
    int grid_period[2];     /* Wraparound */
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;

    int i, j, k;            /* Loop indices */
    dtype* B_T;
    dtype* C_rows;
    dtype* C_rows_interleaved;
    dtype* local_A;
    dtype* local_B_T;
    dtype* local_C;
    int mat_shape[4] = {0};
    int local_rows;         /* Matrix rows on this proc */
    int local_cols;         /* Matrix cols on this proc */
    int* count;
    int* disp;

    if (!id) {
        if (n_A != m_B) {
            printf("Matrix shapes are incompatible\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        transpose(B, &B_T, m_B, n_B);
        // write_matrix(A, m_A, n_A, "A.txt");
        // write_matrix(B, m_B, n_B, "B.txt");
        // write_matrix(B_T, n_B, m_B, "B_T.txt");
        mat_shape[0] = m_A;
        mat_shape[1] = n_A;
        mat_shape[2] = m_B;
        mat_shape[3] = n_B;
    }
    MPI_Bcast(mat_shape, 4, MPI_INT, 0, comm);
    // MPI_Barrier(MPI_COMM_WORLD);
    grid_dim[0] = grid_dim[1] = 0;
    MPI_Dims_create(p, 2, grid_dim);
    
    if (grid_dim[0] > mat_shape[0]) {
        grid_dim[0] = 1;
        grid_dim[1] = 0;
        MPI_Dims_create(p, 2, grid_dim);
        if (grid_dim[1] > mat_shape[3]) {
            if (!id) printf("Too many processors\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    } else if (grid_dim[1] > mat_shape[3]) {
        grid_dim[0] = 0;
        grid_dim[1] = 1;
        MPI_Dims_create(p, 2, grid_dim);
        if (grid_dim[0] > mat_shape[0]) {
            if (!id) printf("Too many processors\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    }
    
    printf("Proc: %d/%d -> [%d, %d]\n", id, p, grid_dim[0], grid_dim[1]);
    grid_period[0] = grid_period[1] = 0;
    MPI_Cart_create(comm, 2, grid_dim, grid_period, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, id, 2, grid_coord);

    // printf("ID: %d -> [%d, %d]\n", id, grid_coord[0], grid_coord[1]);

    MPI_Comm_split(grid_comm, grid_coord[0], grid_coord[1], &row_comm);
    MPI_Comm_split(grid_comm, grid_coord[1], grid_coord[0], &col_comm);

    local_rows = BLOCK_SIZE(grid_coord[0], grid_dim[0], mat_shape[0]);
    local_cols = BLOCK_SIZE(grid_coord[1], grid_dim[1], mat_shape[3]);

    local_A = (dtype*)malloc(local_rows * mat_shape[1] * sizeof(dtype));
    local_B_T = (dtype*)malloc(local_cols * mat_shape[2] * sizeof(dtype));

    // first column, scatter rows of A
    count = NULL;
    disp = NULL;
    if (grid_coord[1] == 0) {
        count = (int*)malloc(grid_dim[0] * sizeof(int));
        disp = (int*)malloc(grid_dim[0] * sizeof(int));
        count[0] = BLOCK_SIZE(0, grid_dim[0], mat_shape[0]) * mat_shape[1];
        disp[0] = 0;
        for (i = 1; i < grid_dim[0]; ++i) {
            disp[i] = disp[i - 1] + count[i - 1];
            count[i] = BLOCK_SIZE(i, grid_dim[0], mat_shape[0]) * mat_shape[1];
        }
        MPI_Scatterv(A, count, disp, mpitype, local_A, count[grid_coord[0]], mpitype, 0, col_comm);
    }

    // first row, scatter cols of B (rows of B^T)
    if (grid_coord[0] == 0) {
        count = (int*)realloc(count, grid_dim[1] * sizeof(int));
        disp = (int*)realloc(disp, grid_dim[1] * sizeof(int));
        count[0] = BLOCK_SIZE(0, grid_dim[1], mat_shape[3]) * mat_shape[2];
        disp[0] = 0;
        for (i = 1; i < grid_dim[1]; ++i) {
            disp[i] = disp[i - 1] + count[i - 1];
            count[i] = BLOCK_SIZE(i, grid_dim[1], mat_shape[3]) * mat_shape[2];
        }
        MPI_Scatterv(B_T, count, disp, mpitype, local_B_T, count[grid_coord[1]], mpitype, 0, row_comm);
    }

    // bcast rows of A
    MPI_Bcast(local_A, local_rows * mat_shape[1], mpitype, 0, row_comm);

    // bcast cols of B
    MPI_Bcast(local_B_T, local_cols * mat_shape[2], mpitype, 0, col_comm);

    // char path_a[64];
    // char path_b[64];
    // sprintf(path_a, "A_%d_%d.txt", grid_coord[0], grid_coord[1]);
    // sprintf(path_b, "B_T_%d_%d.txt", grid_coord[0], grid_coord[1]);
    // write_matrix(local_A, local_rows, mat_shape[1], path_a);
    // write_matrix(local_B_T, local_cols, mat_shape[2], path_b);
    // GEMM
    // printf("ID: %d -> [%d, %d] (%d, %d, %d, %d)\n", id, grid_coord[0], grid_coord[1], local_rows, mat_shape[1], mat_shape[2], local_cols);
    matrix_mul_transposed_inner(local_A, local_B_T, &local_C, local_rows, mat_shape[1], mat_shape[2], local_cols);

    // char path_c[64];
    // sprintf(path_c, "C_%d_%d.txt", grid_coord[0], grid_coord[1]);
    // write_matrix(local_C, local_rows, local_cols, path_c);
    // first col, gather blocks of C
    count = (int*)realloc(count, grid_dim[1] * sizeof(int));
    disp = (int*)realloc(disp, grid_dim[1] * sizeof(int));
    count[0] = local_rows * BLOCK_SIZE(0, grid_dim[1], mat_shape[3]);
    disp[0] = 0;
    for (i = 1; i < grid_dim[1]; ++i) {
        disp[i] = disp[i - 1] + count[i - 1];
        count[i] = local_rows * BLOCK_SIZE(i, grid_dim[1], mat_shape[3]);
    }
    if (grid_coord[1] == 0) {
        C_rows = (dtype*)malloc(local_rows * mat_shape[3] * sizeof(dtype));
    }
    MPI_Gatherv(local_C, count[grid_coord[1]], mpitype, C_rows, count, disp, mpitype, 0, row_comm);

    // Proc (0, 0) gather rows of C
    if (grid_coord[1] == 0) {
        C_rows_interleaved = (dtype*)malloc(local_rows * mat_shape[3] * sizeof(dtype));
        interleave_rows(C_rows, C_rows_interleaved, local_rows, mat_shape[3], grid_dim[1]);
        // sprintf(path_c, "C_int_%d_%d.txt", grid_coord[0], grid_coord[1]);
        // write_matrix(C_rows_interleaved, local_rows, mat_shape[3], path_c);

        count = (int*)realloc(count, grid_dim[0] * sizeof(int));
        disp = (int*)realloc(disp, grid_dim[0] * sizeof(int));
        count[0] = BLOCK_SIZE(0, grid_dim[0], mat_shape[0]) * mat_shape[3];
        disp[0] = 0;
        for (i = 0; i < grid_dim[0]; ++i) {
            disp[i] = disp[i - 1] + count[i - 1];
            count[i] = BLOCK_SIZE(i, grid_dim[0], mat_shape[0]) * mat_shape[3];
        }
        if (grid_coord[0] == 0) {
            *C = (dtype*)malloc(mat_shape[0] * mat_shape[3] * sizeof(dtype));
        }
        MPI_Gatherv(C_rows_interleaved, count[grid_coord[0]], mpitype, *C, count, disp, mpitype, 0, col_comm);
    }

    if (!id) {
        free(B_T);
    }
    if (grid_coord[1] == 0) {
        free(C_rows);
        free(C_rows_interleaved);
    }
    free(local_A);
    free(local_B_T);
    free(local_C);
    free(count);
    free(disp);
}

void gemm(int id, int p, char* path_A, char* path_B, MPI_Comm comm) {
    dtype* A;
    dtype* B;
    dtype* C;
    int m_A, n_A, m_B, n_B;

    if (!id) {
        read_matrix(path_A, &m_A, &n_A, &A);
        read_matrix(path_B, &m_B, &n_B, &B);
    }
    matrix_mul(id, p, A, B, &C, m_A, n_A, m_B, n_B, comm);

    // Write file
    if (!id) {
        write_matrix(C, m_A, n_B, "C.txt");
        write_matrix_bin(C, m_A, n_B, "C");
    }
    if (!id) {
        free(A);
        free(B);
        free(C);
    }
}

void conv(int id, int p, char* path_f, char* path_k, int stride_h, int stride_w, MPI_Comm comm) {
    dtype* feature_map;
    dtype* kernel;
    dtype* A;
    dtype* B;
    dtype* C;
    int h_in, w_in;
    int h_out, w_out;
    int k_h, k_w;
    int m_A, n_A, m_B, n_B;

    if (!id) {
        read_matrix(path_f, &h_in, &w_in, &feature_map);
        read_matrix(path_k, &k_h, &k_w, &kernel);
        im2col_hw(h_in, w_in, k_h, k_w, stride_h, stride_w, feature_map, kernel, &A, &B, &h_out, &w_out, &m_A, &n_A, &m_B, &n_B);
    }
    matrix_mul(id, p, A, B, &C, m_A, n_A, m_B, n_B, comm);
    // Write file
    if (!id) {
        write_matrix(C, h_out, w_out, "Conv.txt");
        write_matrix_bin(C, h_out, w_out, "Conv");
    }
    if (!id) {
        free(feature_map);
        free(kernel);
        free(A);
        free(C);
    }
}

void avgpooling(int id, int p, char* path_f, int stride_h, int stride_w, int k_h, int k_w, MPI_Comm comm) {
    dtype* feature_map;
    dtype* kernel;
    dtype* A;
    dtype* B;
    dtype* C;
    int h_in, w_in;
    int h_out, w_out;
    int m_A, n_A, m_B, n_B;
    int i;

    if (!id) {
        read_matrix(path_f, &h_in, &w_in, &feature_map);
        kernel = (dtype*)malloc(k_h * k_w * sizeof(dtype));
        if (mpitype == MPI_INT) {
            for (i = 0; i < k_h * k_w; ++i) {
                kernel[i] = 1;
            }
        } else {
            for (i = 0; i < k_h * k_w; ++i) {
                kernel[i] = (dtype)1 / (k_h * k_w);
            }
        }
        im2col_hw(h_in, w_in, k_h, k_w, stride_h, stride_w, feature_map, kernel, &A, &B, &h_out, &w_out, &m_A, &n_A, &m_B, &n_B);
    }
    matrix_mul(id, p, A, B, &C, m_A, n_A, m_B, n_B, comm);
    // Write file
    if (mpitype == MPI_INT) {
        for (i = 0; i < h_out * w_out; ++i) {
            C[i] = (int)C[i] / (k_h * k_w);
        }
    }
    if (!id) {
        write_matrix(C, h_out, w_out, "AvgPooling.txt");
        write_matrix_bin(C, h_out, w_out, "AvgPooling");
    }
    if (!id) {
        free(feature_map);
        free(kernel);
        free(A);
        free(C);
    }
}

int main(int argc, char *argv[]) {
    int id;                /* Process rank */
    int p;                 /* Number of processes */
    int mode;
    int k_h, k_w;
    int stride_h, stride_w;
    double elapsed_time;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    mode = atoi(argv[1]);
    if (mode != 0 && mode != 1 && mode != 2) {
        if (!id) printf("mode = 0: GEMM, mode = 1: Conv via GEMM, mode = 2: AvgPooling via GEMM\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();
    if (mode == 0) {
        if (argc != 4) {
            if (!id) printf("GEMM: %s 0 <path_A> <path_B>\n", argv[0]);
            MPI_Finalize();
            exit(1);
        }
        // GEMM
        gemm(id, p, argv[2], argv[3], MPI_COMM_WORLD);
    } else if (mode == 1) {
        if (argc != 6) {
            if (!id) printf("Conv: %s 1 <path_f> <path_k> <stride_h> <stride_w>\n", argv[0]);
            MPI_Finalize();
            exit(1);
        } else {
            stride_h = atoi(argv[4]);
            stride_w = atoi(argv[5]);
        }
        // Conv
        conv(id, p, argv[2], argv[3], stride_h, stride_w, MPI_COMM_WORLD);
    } else {
        if (argc != 7) {
            if (!id) printf("AvgPooling: %s 2 <path_f> <stride_h> <stride_w> <k_h> <k_w>\n", argv[0]);
            MPI_Finalize();
            exit(1);
        } else {
            stride_h = atoi(argv[3]);
            stride_w = atoi(argv[4]);
            k_h = atoi(argv[5]);
            k_w = atoi(argv[6]);
        }
        // AvgPooling
        avgpooling(id, p, argv[2], stride_h, stride_w, k_h, k_w, MPI_COMM_WORLD);
    }
    elapsed_time += MPI_Wtime();
    // if (!id) {
        printf("ID: %d Total elapsed time: %10.6f\n", id, elapsed_time);
        fflush(stdout);
    // }

    MPI_Finalize();
    return 0;
}