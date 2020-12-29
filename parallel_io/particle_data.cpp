#include "particle_data.h"

WritePerformance::WritePerformance(int mpi_rank, int mpi_size, uint64_t local_particle_count)
    : mpi_rank(mpi_rank), mpi_size(mpi_size), local_particle_count(local_particle_count)
{
}
