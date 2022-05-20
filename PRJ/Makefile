CC=gcc
MPI_CC=mpicc

all: build-prj

build-prj:
	make 1-1 1-2 1-3 2-1 2-2 2-3

1-1: 1-1.c
	$(MPI_CC) $^ -o $@ -lm

1-2: 1-2.c
	$(MPI_CC) $^ -o $@ -lm

1-3: 1-3.c
	$(MPI_CC) $^ -o $@ -lm

2-1: 2-1.c
	$(CC) $^ -o $@ -fopenmp -lm

2-2: 2-2.c
	$(CC) $^ -o $@ -fopenmp -lm

2-3: 2-3.c
	$(CC) $^ -o $@ -fopenmp -lm

clean:
	rm -f 1-1 1-2 1-3 2-1 2-2 2-3