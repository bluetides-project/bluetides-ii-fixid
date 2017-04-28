#include <stdio.h>
#include <mpi.h>
#include <bigfile.h>
#include <bigfile-mpi.h>
#include <mpsort.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "endrun.h"

typedef struct {
    uint64_t oldid;
    uint64_t id;
    uint64_t rank;
    int ptype;
    char generation;
    char oldgeneration;
    float sft;
} dtype;

static void fixid(char * fname);
static uint64_t
MPI_cumsum(uint64_t a, MPI_Comm comm);

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    char * fname = argv[1];

    fixid(fname);

    MPI_Finalize();
}

static void
radix_by_id(const void * ptr, void * radix, void * arg)
{
    const dtype * P = (const dtype*) ptr;
    memcpy(radix, &P->id, 8);
}
static void
radix_by_rank(const void * ptr, void * radix, void * arg)
{
    const dtype * P = (const dtype*) ptr;
    memcpy(radix, &P->rank, 8);
}

static int
cmp_by_sft(const void * ptr1, const void * ptr2)
{
    const dtype * P1 = (const dtype*) ptr1;
    const dtype * P2 = (const dtype*) ptr2;
    if(P1->sft > P2->sft) return 1;
    if(P1->sft == P2->sft) return 0;
    if(P1->sft < P2->sft) return -1;
}
void
fix_one(dtype * P, uint64_t size)
{
    if(size == 1) {
        P[0].generation = 0;
        return;
    }
    if(size > 10) {
        endrun(1, "too many particles with the same id, something is wrong.\n");
    }
    qsort(P, size, sizeof(P[0]), cmp_by_sft);
    int i;

    int N[6] = {0};
    for(i = 0; i < size; i ++) {
        N[P[i].ptype] ++;
        P[i].generation = i + 1;
    }
    if(N[0] > 1) {
        endrun(1, "too many gas particles with the same id, something is wrong. N[0] = %d\n", N[0]);
    }
    if(N[1] > 0) {
        endrun(1, "too many dm particles with the same id, something is wrong. N[1] = %d\n", N[1]);
    }
    if(P[size - 1].ptype != 0) {
        endrun(1, "last particle is not gas, something is wrong. ptype %d\n", P[size-1].ptype);
    }
    P[size - 1].generation = size - 1;
    uint64_t idgroup = P[0].id;
    for(i = 0; i < size; i ++) {
        if(P[i].ptype != 0) {
            uint64_t g = P[i].generation;
            P[i].id += (g << 56L);
        }
    }
    message(1, "fix id group %08lX Ngas = %d Nstar=%d, Nbh=%d\n", idgroup, N[0], N[4], N[5]);
}

void fixid(char * fname)
{
    int ThisTask;
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    int NFileID;
    int NFileGeneration;

    BigFile bf[1] = {0};
    BigBlock bb[1] = {0};

    dtype * P = NULL;
    dtype * Q0 = NULL;
    dtype * Q = NULL;

    if (0 != big_file_mpi_open(bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "failed open file %s\n", fname);
    }

    uint64_t TotNumPart[6];
    uint64_t Start[6];
    uint64_t End[6];
    uint64_t LocalN = 0;
    big_file_mpi_open_block(bf, bb, "Header", MPI_COMM_WORLD);
    big_block_get_attr(bb, "TotNumPart", TotNumPart, "u8", 6);
    big_block_mpi_close(bb, MPI_COMM_WORLD);

    int ptype;

    for(ptype = 0; ptype < 6; ptype ++) {
        Start[ptype] =TotNumPart[ptype] * ThisTask / NTask;
        End [ptype] = TotNumPart[ptype] * (ThisTask + 1)/ NTask;
        LocalN += (End[ptype] - Start[ptype]);
        message(0, "Rank 0: Start[%d]:End[%d] is %lu:%lu \n", ptype, ptype, Start[ptype], End[ptype]);
    }

    uint64_t LocalOffset = MPI_cumsum(LocalN, MPI_COMM_WORLD);

    message(0, "Rank 0 : LocalN is %lu, LocalOffset is %lu\n", LocalN, LocalOffset);

    /* max eq group is 255 particles */
    P = malloc(sizeof(P[0]) * (LocalN + 255));
    Q0 = malloc(sizeof(P[0]) * (LocalN + 255));
    Q = P;

    for(ptype = 0; ptype < 6; ptype ++) {
        if(TotNumPart[ptype] == 0) continue;
        char blockname[80];
        BigArray array[1];
        BigBlockPtr ptr[1];
        int64_t i;

        sprintf(blockname, "%d/ID.broken", ptype);
        if(0 == big_file_mpi_open_block(bf, bb, blockname, MPI_COMM_WORLD)) {
            big_array_init(array, &Q[0].id, "u8", 1, (size_t []) {End[ptype] - Start[ptype]}, (ptrdiff_t []) {sizeof(P[0])});
            big_block_seek(bb, ptr, 0);
            big_block_mpi_read(bb, ptr, array, 0, MPI_COMM_WORLD);
            NFileID = bb->Nfile;
            big_block_mpi_close(bb, MPI_COMM_WORLD);
        } else {
            endrun(0, "failed to read ID.broken block, ptype = %d\n", ptype);
        }
        message(0, "Finished reading %s\n", blockname);
        sprintf(blockname, "%d/Generation.broken", ptype);
        if(0 == big_file_mpi_open_block(bf, bb, blockname, MPI_COMM_WORLD)) {
            /* the old generation is not really used, but we read anyways to ensure the back exists */
            big_array_init(array, &Q[0].oldgeneration, "u1", 1, (size_t []) {End[ptype] - Start[ptype]}, (ptrdiff_t []) {sizeof(P[0])});
            big_block_seek(bb, ptr, 0);
            big_block_mpi_read(bb, ptr, array, 0, MPI_COMM_WORLD);
            NFileGeneration = bb->Nfile;
            big_block_mpi_close(bb, MPI_COMM_WORLD);

        } else {
            endrun(0, "failed to read Generation.broken block, ptype = %d\n", ptype);
        }
        message(0, "Finished reading %s\n", blockname);
        sprintf(blockname, "%d/StarFormationTime", ptype);
        if(0 == big_file_mpi_open_block(bf, bb, blockname, MPI_COMM_WORLD)) {
            big_array_init(array, &Q[0].sft, "f4", 1, (size_t []) {End[ptype] - Start[ptype]}, (ptrdiff_t []) {sizeof(P[0])});
            big_block_seek(bb, ptr, 0);
            big_block_mpi_read(bb, ptr, array, 0, MPI_COMM_WORLD);
            big_block_mpi_close(bb, MPI_COMM_WORLD);
        } else {
            for(i = 0; i < End[ptype] - Start[ptype]; i ++) {
                /* no sft attribute, move to the end. */
                Q[i].sft = 100.;
            }
        }
        message(0, "Finished reading %s\n", blockname);
        for(i = 0; i < End[ptype] - Start[ptype]; i ++) {
            Q[i].oldid = Q[i].id;
            Q[i].id = Q[i].id & 0xffffffffffffff; /* keep the lower 12 bytes */
            Q[i].ptype = ptype;
        }
        Q += End[ptype] - Start[ptype];
    }
    int64_t i;
    for(i = 0; i < LocalN; i ++) {
        P[i].rank = i + LocalOffset;
    }

    uint64_t LocalN1 = LocalN;
    message(0, "Beging sorting \n");

    Q = Q0;
    mpsort_mpi_newarray(P, LocalN,
                Q, LocalN1, sizeof(P[0]), radix_by_id, 8, NULL, MPI_COMM_WORLD);

    message(0, "Finished sorting \n");
    /* send the first eqv group to the previous rank in case it is actually part of the previous rank. */
    uint64_t id0 = Q[0].id;
    /* but do not send anything from the first rank */
    for(i = 0; i < LocalN1 && LocalOffset != 0; i ++) {
        if(Q[i].id != id0) break;
    }
    int Nsend = i;
    int Nrecv;
    MPI_Sendrecv(&Nsend, 1, MPI_INT, (NTask + ThisTask - 1) % NTask, 0,
                 &Nrecv, 1, MPI_INT, (ThisTask + 1) % NTask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(Q, sizeof(P[0]) * Nsend, MPI_BYTE, (NTask + ThisTask - 1) % NTask, 1,
                 Q + LocalN1, sizeof(P[0]) * Nrecv, MPI_BYTE, (ThisTask + 1) % NTask, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    LocalN1 += Nrecv;
    LocalN1 -= Nsend;
    /* Q is the head of the first eqv group. */
    Q += Nsend;

    uint64_t j = 0;
    for(i = 0; i <= LocalN1; i ++) {
        if(i == LocalN1 || Q[i].id != Q[j].id) {
            fix_one(&Q[j], i - j);
            j = i;
        }
    }

    /* return the particles to P */
    message(0, "Beging sorting \n");
    mpsort_mpi_newarray(Q, LocalN1, P, LocalN, sizeof(P[0]), radix_by_rank, 8, NULL, MPI_COMM_WORLD);
    message(0, "Finished sorting \n");
    for(i = 0; i <= LocalN; i ++) {
        if(P[i].oldid != P[i].id || P[i].oldgeneration != P[i].generation) {
            message(1, "old id %08lX new id %08lX, ptype=%d, oldgen=%d, newgen=%d\n", P[i].oldid, P[i].id, P[i].ptype, P[i].oldgeneration, P[i].generation);
        }
    }
    /* write */
    Q = P;
    for(ptype = 0; ptype < 6; ptype ++) {
        if(TotNumPart[ptype] == 0) continue;
        char blockname[80];
        BigArray array[1];
        BigBlockPtr ptr[1];
        int64_t i;

        sprintf(blockname, "%d/ID", ptype);
        if(0 == big_file_mpi_create_block(bf, bb, blockname, "u8", 1, NFileID, TotNumPart[ptype], MPI_COMM_WORLD)) {
            big_array_init(array, &Q[0].id, "u8", 1, (size_t []) {End[ptype] - Start[ptype]}, (ptrdiff_t []) {sizeof(P[0])});
            big_block_seek(bb, ptr, 0);
            big_block_mpi_write(bb, ptr, array, 0, MPI_COMM_WORLD);
            big_block_mpi_close(bb, MPI_COMM_WORLD);
        } else {
            endrun(0, "failed to write ID block, ptype = %d\n", ptype);
        }
        message(0, "Finished writing %s\n", blockname);
        sprintf(blockname, "%d/Generation", ptype);
        if(0 == big_file_mpi_create_block(bf, bb, blockname, "u1", 1, NFileGeneration, TotNumPart[ptype], MPI_COMM_WORLD)) {
            /* the old generation is not really used, but we read anyways to ensure the back exists */
            big_array_init(array, &Q[0].generation, "u1", 1, (size_t []) {End[ptype] - Start[ptype]}, (ptrdiff_t []) {sizeof(P[0])});
            big_block_seek(bb, ptr, 0);
            big_block_mpi_write(bb, ptr, array, 0, MPI_COMM_WORLD);
            big_block_mpi_close(bb, MPI_COMM_WORLD);

        } else {
            endrun(0, "failed to read Generation block, ptype = %d\n", ptype);
        }
        message(0, "Finished writing %s\n", blockname);
        Q += End[ptype] - Start[ptype];
    }
    free(Q0);
    free(P);
}

static uint64_t
MPI_cumsum(uint64_t a, MPI_Comm comm)
{
    int size;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    uint64_t * gather = malloc(sizeof(uint64_t) * size);
    MPI_Allgather(&a, 1, MPI_LONG_LONG, gather, 1, MPI_LONG_LONG, comm);
    uint64_t r = 0;
    int i;
    for(i = 0; i < rank; i ++) {
        r += gather[i];
    }
    return r;
}
