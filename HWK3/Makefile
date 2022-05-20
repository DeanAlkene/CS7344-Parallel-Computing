CC=mpicc

all: build-hwk3

build-hwk3:
	make 8-6 8-12 9-7 9-10

8-6: 8-6.c MyMPI.c
	$(CC) $^ -o $@ -lm

8-12: 8-12-base 8-12-elem 8-12-diag-col

8-12-base: 8-12-base.c MyMPI.c
	$(CC) $^ -o $@ -lm

8-12-elem: 8-12-elem.c MyMPI.c
	$(CC) $^ -o $@ -lm

8-12-diag-col: 8-12-diag-col.c MyMPI.c
	$(CC) $^ -o $@ -lm

9-7: 9-7.c MyMPI.c
	$(CC) $^ -o $@ -lm

9-10: 9-10.c MyMPI.c
	$(CC) $^ -o $@ -lm

clean:
	rm -f 8-6 8-12-base 8-12-elem 8-12-diag-col 9-7 9-10