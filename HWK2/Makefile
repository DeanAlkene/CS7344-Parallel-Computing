CC=mpicc

all: build-hwk2

build-hwk2:
	make 5-9-base 5-9 6-10 6-13

5-9-base: 5-9-base.c
	$(CC) $^ -o $@ -lm

5-9: 5-9.c
	$(CC) $^ -o $@ -lm

6-10: 6-10.c
	$(CC) $^ -o $@ -lm

6-13: 6-13.c MyMPI.c
	$(CC) $^ -o $@ -lm

clean:
	rm -f 5-9-base 5-9 6-10 6-13
