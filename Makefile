MPICC = mpicc
LDFLAGS = -L bigfile/src -L mpsort/
CFLAGS = -I bigfile/src -I mpsort/
main: main.c endrun.c endrun.h
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o $@ main.c endrun.c -lbigfile-mpi -lbigfile -lmpsort-mpi -lradixsort

clean:
	rm -rf main *.o
