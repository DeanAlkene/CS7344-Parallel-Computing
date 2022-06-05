#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define WORD_LEN 256
#define HTABLE_SIZE (1 << 15)
#define MAX_LINE_NUM 30000
#define LINE_LEN 128
#define WORD_COUNT_TAG 0
#define STRING_TAG 1

#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

struct WordCount {
    int count;
    char word[WORD_LEN];
};

struct Node {
    struct WordCount wordcount;
    struct Node* next;
};

struct HashTable {
    struct Node** htable;
    int size;
};

// Hash function
unsigned int MurmurOAAT32(const char* key) {
    unsigned int h = 3323198485ul;
    for (;*key;++key) {
        h ^= *key;
        h *= 0x5bd1e995;
        h ^= h >> 15;
    }
    return h;
}

struct Node* find_word(struct HashTable* htable, char* word) {
    unsigned int hash;
    struct Node *head, *cur;

    hash = MurmurOAAT32(word) % HTABLE_SIZE;
    head = htable->htable[hash];

    for (cur = head; cur != NULL; cur = cur->next) {
        if (strcmp(word, cur->wordcount.word) == 0) {
            return cur;
        }
    }
    return NULL;
}

void add_word_count(struct HashTable* htable, char* word, int count) {
    unsigned int hash;
    struct Node *head, *cur;
    
    hash = MurmurOAAT32(word) % HTABLE_SIZE;
    head = htable->htable[hash];

    for (cur = head; cur != NULL; cur = cur->next) {
        if (strcmp(word, cur->wordcount.word) == 0) {
            // update count
            cur->wordcount.count += count;
            return;
        }
    }

    // insert a new node before head
    cur = (struct Node*)malloc(sizeof(struct Node));
    strcpy(cur->wordcount.word, word);
    cur->wordcount.count = count;
    cur->next = head;
    htable->htable[hash] = cur;
    htable->size++;
}

void clear_htable(struct HashTable* htable) {
    struct Node *cur, *next;
    int i;
    for (i = 0; i < HTABLE_SIZE; ++i) {
        cur = htable->htable[i];
        while (cur != NULL) {
            next = cur->next;
            free(cur);
            cur = next;
        }
    }
}

void count_lines(int p, char* file_buf, int buf_size, size_t* disp) {
    int i, lines;
    size_t line_idx[MAX_LINE_NUM];
    
    lines = 0;
    line_idx[0] = 0;
    // count lines of a file
    for (i = 0; i < buf_size; ++i) {
        if (file_buf[i] == '\n') {
            line_idx[lines++] = i;
        }
    }
    // if the file is not ended with \n
    if (file_buf[buf_size - 1] != '\n') {
        line_idx[lines++] = buf_size - 1;
    }
    // generate a displacement array for transmission
    disp[0] = 0;
    for (i = 0; i < p; ++i) {
        disp[i + 1] = line_idx[BLOCK_HIGH(i, p, lines)] + 1;
    }
}

void write_file(struct HashTable* htable, char* path) {
    FILE* f;
    int i;
    struct Node* cur;

    f = fopen(path, "w");
    for (i = 0; i < HTABLE_SIZE; ++i) {
        for (cur = htable->htable[i]; cur != NULL; cur = cur->next) {
            fprintf(f, "%s %d\n", cur->wordcount.word, cur->wordcount.count);
        }
    }
    fclose(f);
}

void serialize(struct HashTable* htable, struct WordCount** buf) {
    int i, j;
    struct Node* cur;

    j = 0;
    *buf = (struct WordCount*)malloc(htable->size * sizeof(struct WordCount));
    // serialize a HashTable to a buffer
    for (i = 0; i < HTABLE_SIZE; ++i) {
        for (cur = htable->htable[i]; cur != NULL; cur = cur->next) {
            strcpy((*buf)[j].word, cur->wordcount.word);
            (*buf)[j].count = cur->wordcount.count;
            j++;
        }
    }
}

void map_small(int id, int p, int n, char* path, struct HashTable** htable) {
    FILE* f;                /* File pointer */
    char* word;             /* Word */
    char* read_buf;         /* Buffer to store the file */
    int total_size;         /* The size of the buffer */
    char file_path[64];     /* File path */
    struct stat fstat;      /* Stat of a file */
    int i;                  /* Loop index */

    *htable = (struct HashTable*)malloc(sizeof(struct HashTable));
    (*htable)->size = 0;
    (*htable)->htable = (struct Node**)malloc(HTABLE_SIZE * sizeof(struct Node*));
    memset((*htable)->htable, 0, HTABLE_SIZE * sizeof(struct Node*));
    // open files & count (map + combine)
    for (i = id; i < n; i += p) {
        sprintf(file_path, "%s/", path);
        sprintf(file_path + strlen(file_path), "small_%d.txt", i + 100);
        f = fopen(file_path, "r");
        stat(file_path, &fstat);
        total_size = fstat.st_size;
        read_buf = (char*)malloc((total_size + 1) * sizeof(char));
        fread(read_buf, 1, total_size, f);
        fclose(f);

        read_buf[total_size] = '\0';
        // word = strtok(read_buf, " ,.-\n:!?()';\"\t\r");
        // naive parse
        word = strtok(read_buf, " \t\r\n");
        while (word != NULL) {
            // add to the HashTable
            add_word_count(*htable, word, 1);
            // word = strtok(NULL, " ,.-\n:!?()';\"\t\r");
            word = strtok(NULL, " \t\r\n");
        }
        free(word);
    }
}

void map_big(int id, int p, char* path, struct HashTable** htable) {
    FILE* f;                /* File pointer */
    char* word;             /* Word */
    char file_path[64];     /* File path */
    struct stat fstat;      /* Stat of a file */
    char* read_buf;         /* Buffer to store the file */
    size_t total_size;      /* The size of the buffer */
    int i;                  /* Loop index */

    char *in_buf;           /* Recv buffer */
    int recv_size;          /* Recv size */
    size_t* disp;           /* For send */
    MPI_Status status;      /* Recv status */
    MPI_Request* requests;  /* Send requests */

    // open files & distribute lines (map)
    if (!id) {
        disp = (size_t*)malloc((p + 1) * sizeof(size_t));
        sprintf(file_path, "%s/", path);
        sprintf(file_path + strlen(file_path), "big_100.txt");
        
        f = fopen(file_path, "r");
        stat(file_path, &fstat);
        total_size = fstat.st_size;
        read_buf = (char*)malloc((total_size + 1) * sizeof(char));
        fread(read_buf, 1, total_size, f);
        fclose(f);

        read_buf[total_size] = '\0';
        // generate disp by scanning \n in the buffer
        count_lines(p, read_buf, total_size, disp);

        // distribute lines to other procs
        requests = (MPI_Request*)malloc((p - 1) * sizeof(MPI_Request));
        for (i = 1; i < p; ++i) {
            MPI_Isend(read_buf + disp[i], disp[i + 1] - disp[i], MPI_CHAR, i, STRING_TAG, MPI_COMM_WORLD, &requests[i - 1]);
        }
        // local copy
        recv_size = disp[1];
        in_buf = (char*)malloc((recv_size + 1) * sizeof(char));
        memcpy(in_buf, read_buf, recv_size * sizeof(char));
        in_buf[recv_size] = '\0';
    } else {
        // receive lines for proc 0
        MPI_Probe(0, STRING_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_CHAR, &recv_size);
        in_buf = (char*)malloc((recv_size + 1) * sizeof(char));
        MPI_Recv(in_buf, recv_size, MPI_CHAR, status.MPI_SOURCE, STRING_TAG, MPI_COMM_WORLD, &status);
        in_buf[recv_size] = '\0';
    }

    *htable = (struct HashTable*)malloc(sizeof(struct HashTable));
    (*htable)->size = 0;
    (*htable)->htable = (struct Node**)malloc(HTABLE_SIZE * sizeof(struct Node*));
    memset((*htable)->htable, 0, HTABLE_SIZE * sizeof(struct Node*));
    // count (map + combine)
    // word = strtok(in_buf, " ,.-\n:!?()';\"\t\r");
    // naive parse
    word = strtok(in_buf, " \t\r\n");
    while (word != NULL) {
        // add to the HashTable
        add_word_count(*htable, word, 1);
        // word = strtok(NULL, " ,.-\n:!?()';\"\t\r");
        word = strtok(NULL, " \t\r\n");
    }
    free(word);

    if (!id) {
        // wait for all send requests
        for (i = 1; i < p; ++i) {
            MPI_Wait(&requests[i - 1], &status);
        }
        free(requests);
        free(disp);
        free(read_buf);
    }
    free(in_buf);
}

void reduce_inner(struct HashTable* htable, struct WordCount* buf, int buf_size) {
    int i, j;

    // combine the WordCount buf to the HashTable
    for (i = 0; i < buf_size; ++i) {
        add_word_count(htable, buf[i].word, buf[i].count);
    }
}

void reduce(int id, int p, struct HashTable* htable, int mode) {
    struct WordCount *in_buf, *out_buf; /* Send and receive buffers */
    MPI_Request pending;                /* Send request */
    MPI_Status status;                  /* Recv status */
    int recv_size;                      /* Receive size */
    int terminated;                     /* Terminated procs */

    // for creating the new type corresponding to struct WordCount
    int blocks[2] = {1, WORD_LEN};
    MPI_Datatype types[2] = {MPI_INT, MPI_CHAR};
    MPI_Aint disp[2];
    MPI_Datatype wc_type;
    MPI_Aint lb, int_ext;

    // create a new type corresponding to struct WordCount
    MPI_Type_get_extent(MPI_INT, &lb, &int_ext);
    disp[0] = (MPI_Aint)0;
    disp[1] = int_ext;
    MPI_Type_create_struct(2, blocks, disp, types, &wc_type);
    MPI_Type_commit(&wc_type);

    // Num of reducer = 1, let proc 0 be the reducer
    if (!id) {
        terminated = 0;
        while (terminated < p - 1) {
            // receive from other procs and reduce
            MPI_Probe(MPI_ANY_SOURCE, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, wc_type, &recv_size);
            in_buf = (struct WordCount*)malloc(recv_size * sizeof(struct WordCount));
            MPI_Recv(in_buf, recv_size, wc_type, status.MPI_SOURCE, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
            reduce_inner(htable, in_buf, recv_size);
            free(in_buf);
            terminated++;
        }
        // write file
        if (mode == 0) {
            write_file(htable, "wordcount_small.txt");
        } else {
            write_file(htable, "wordcount_big.txt");
        }
        clear_htable(htable);
        free(htable);
    } else {
        out_buf = (struct WordCount*)malloc(htable->size * sizeof(struct WordCount));
        // serialize local HashTable to WordCount buf
        serialize(htable, &out_buf);
        // MPI_Isend(out_buf, htable->size, wc_type, 0, WORD_COUNT_TAG, MPI_COMM_WORLD, &pending);
        // send to the reducer
        MPI_Send(out_buf, htable->size, wc_type, 0, WORD_COUNT_TAG, MPI_COMM_WORLD);
        free(out_buf);
        clear_htable(htable);
        free(htable);
    }
}

void sharding(int reduce_p, struct HashTable* htable, struct WordCount** buf, size_t* disp) {
    int i, j;
    int low, high;
    struct Node* cur;
    int id;

    j = 0;
    id = 0;
    disp[0] = 0;
    high = BLOCK_HIGH(id, reduce_p, HTABLE_SIZE);
    *buf = (struct WordCount*)malloc(htable->size * sizeof(struct WordCount));
    for (i = 0; i < HTABLE_SIZE; ++i) {
        // serialize the HashTable to a WordCount buf
        for (cur = htable->htable[i]; cur != NULL; cur = cur->next) {
            strcpy((*buf)[j].word, cur->wordcount.word);
            (*buf)[j].count = cur->wordcount.count;
            j++;
        }
        // mark the end of the block that will be send to reducer id
        if (i == high) {
            disp[id + 1] = j;
            high = BLOCK_HIGH(++id, reduce_p, HTABLE_SIZE);
        }
    }
}

void write_partial_file(int id, int reduce_p, struct HashTable* htable, char* name) {
    FILE* f;
    int i, low, high;
    struct Node* cur;
    char file_path[64];

    if (id >= reduce_p) return;

    sprintf(file_path, "%s_%d.txt", name, id);
    f = fopen(file_path, "w");
    low = BLOCK_LOW(id, reduce_p, HTABLE_SIZE);
    high = BLOCK_HIGH(id, reduce_p, HTABLE_SIZE);
    for (i = low; i <= high; ++i) {
        for (cur = htable->htable[i]; cur != NULL; cur = cur->next) {
            fprintf(f, "%s %d\n", cur->wordcount.word, cur->wordcount.count);
        }
    }
    fclose(f);
}

void reduce_with_shuffle(int id, int p, int reduce_p, struct HashTable* htable, int mode) {
    struct WordCount *in_buf, *out_buf; /* Send and receive buffers */
    int i;                              /* Loop index */
    int send_size, recv_size;           /* Send and recv sizes */
    int terminated;                     /* Terminated procs */
    size_t* send_disp;                  /* For sending blocks of buf */
    MPI_Request* requests;              /* Send request */
    MPI_Status status;                  /* Receive status */

    // for creating the new type corresponding to struct WordCount
    int blocks[2] = {1, WORD_LEN};
    MPI_Datatype types[2] = {MPI_INT, MPI_CHAR};
    MPI_Aint disp[2];
    MPI_Datatype wc_type;
    MPI_Aint lb, int_ext;

    // create a new type corresponding to struct WordCount
    MPI_Type_get_extent(MPI_INT, &lb, &int_ext);
    disp[0] = (MPI_Aint)0;
    disp[1] = int_ext;
    MPI_Type_create_struct(2, blocks, disp, types, &wc_type);
    MPI_Type_commit(&wc_type);

    // sharding: serialize HashTable and calculate a displacement array for sending
    send_disp = (size_t*)malloc((reduce_p + 1) * sizeof(size_t));
    sharding(reduce_p, htable, &out_buf, send_disp);
    requests = (MPI_Request*)malloc(reduce_p * sizeof(MPI_Request));
    // send blocks of WordCount buf to the corresponding proc
    for (i = 0; i < reduce_p; ++i) {
        if (i != id) {
            send_size = send_disp[i + 1] - send_disp[i];
            MPI_Isend(out_buf + send_disp[i], send_size, wc_type, i, WORD_COUNT_TAG, MPI_COMM_WORLD, &requests[i]);
        }
    }

    // multiple reducers
    if (id < reduce_p) {
        terminated = 0;
        // receive WordCount from p - 1 procs
        while (terminated < p - 1) {
            MPI_Probe(MPI_ANY_SOURCE, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, wc_type, &recv_size);
            in_buf = (struct WordCount*)malloc(recv_size * sizeof(struct WordCount));
            MPI_Recv(in_buf, recv_size, wc_type, status.MPI_SOURCE, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
            // reduce all received WordCounts
            reduce_inner(htable, in_buf, recv_size);
            free(in_buf);
            terminated++;
        }
        if (mode == 0) {
            write_partial_file(id, reduce_p, htable, "wordcount_small");
        } else {
            write_partial_file(id, reduce_p, htable, "wordcount_big");
        }
    }
    // wait for Isend
    for (i = 0; i < reduce_p; ++i) {
        if (i != id) {
            MPI_Wait(&requests[i], &status);
        }
    }
    clear_htable(htable);
    free(htable);
    free(requests);
    free(out_buf);
    free(send_disp);
}

void reduce_multi_stage(int id, int p, struct HashTable* htable, int mode) {
    struct WordCount *in_buf, *out_buf; /* Send and receive buffers */
    int i;                              /* Loop index */
    int send_size, recv_size;           /* Send and recv sizes */
    int participants;                   /* Active proc in a reduction stage */
    int cur_part;                       /* Number of reducers in a reduction stage */
    MPI_Status status;                  /* Recv status */

    // for creating the new type corresponding to struct WordCount
    int blocks[2] = {1, WORD_LEN};
    MPI_Datatype types[2] = {MPI_INT, MPI_CHAR};
    MPI_Aint disp[2];
    MPI_Datatype wc_type;
    MPI_Aint lb, int_ext;

    // create a new type corresponding to struct WordCount
    MPI_Type_get_extent(MPI_INT, &lb, &int_ext);
    disp[0] = (MPI_Aint)0;
    disp[1] = int_ext;
    MPI_Type_create_struct(2, blocks, disp, types, &wc_type);
    MPI_Type_commit(&wc_type);

    participants = p;
    while (participants > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        // number of reducers in current turn
        cur_part = participants / 2 + (participants % 2);
        if (id < cur_part) {
            // reducers receive WordCounts from their corresponding non-reducers
            if (id + cur_part < participants) {
                MPI_Probe(id + cur_part, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, wc_type, &recv_size);
                in_buf = (struct WordCount*)malloc(recv_size * sizeof(struct WordCount));
                MPI_Recv(in_buf, recv_size, wc_type, status.MPI_SOURCE, WORD_COUNT_TAG, MPI_COMM_WORLD, &status);
                // reduce
                reduce_inner(htable, in_buf, recv_size);
                free(in_buf);
            }
        }
        if (id >= cur_part && id < participants) {
            out_buf = (struct WordCount*)malloc(htable->size * sizeof(struct WordCount));
            // serialize the HashTable and send to reducers
            serialize(htable, &out_buf);
            MPI_Send(out_buf, htable->size, wc_type, id - cur_part, WORD_COUNT_TAG, MPI_COMM_WORLD);
            free(out_buf);
        }
        // number of active procs in the next turn
        participants = cur_part;
    }

    if (!id) {
        if (mode == 0) {
            write_file(htable, "wordcount_small.txt");
        } else {
            write_file(htable, "wordcount_big.txt");
        }
    }
    
    clear_htable(htable);
    free(htable);
}

void word_count_small(int id, int p, int reduce_p, char *path, int n) {
    struct HashTable* htable;

    if (n < p) {
        if (!id) printf("Too many processors!\n");
        MPI_Finalize();
        exit(1);
    }

    map_small(id, p, n, path, &htable);
    if (reduce_p == 0) {
        reduce_multi_stage(id, p, htable, 0);
    } else if (reduce_p == 1) {
        reduce(id, p, htable, 0);
    } else {
        reduce_with_shuffle(id, p, reduce_p, htable, 0);
    }
}

void word_count_big(int id, int p, int reduce_p, char *path) {
    struct HashTable* htable;

    map_big(id, p, path, &htable);
    if (reduce_p == 0) {
        reduce_multi_stage(id, p, htable, 1);
    } else if (reduce_p == 1) {
        reduce(id, p, htable, 1);
    } else {
        reduce_with_shuffle(id, p, reduce_p, htable, 1);
    }
}

int main(int argc, char *argv[]) {
    int id;                 /* Process rank */
    int p;                  /* Number of processes */
    int n;                  /* Number of files */
    int mode;               /* 0 - small files, 1 - big file */
    int reduce_p;           /* 0 - multi-stage, o.w. number of reducers*/
    double elapsed_time;    /* Elapsed time */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 5) {
        if (!id) printf("Command line: %s <mode> <num_reducer> <path> <n>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    mode = atoi(argv[1]);
    if (mode != 0 && mode != 1) {
        if (!id) printf("mode = 0: small files, mode = 1: big file\n");
        MPI_Finalize();
        exit(1);
    }
    reduce_p = atoi(argv[2]);
    if (reduce_p < 0 || reduce_p > p) {
        if (!id) printf("1 <= num_reducer <= p, 0 for multi-stage reduction\n");
        MPI_Finalize();
        exit(1);
    }
    n = atoi(argv[4]);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = 0.0;
    elapsed_time -= MPI_Wtime();
    if (!mode) {
        // small files
        word_count_small(id, p, reduce_p, argv[3], n);
    } else {
        // big file
        word_count_big(id, p, reduce_p, argv[3]);
    }
    elapsed_time += MPI_Wtime();
    if (!id) {
        printf("Elapsed time: %10.3f ms\n", elapsed_time * 1000);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}