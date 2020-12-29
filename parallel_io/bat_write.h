#pragma once

#include <stdint.h>
#include "bat_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *BATParticleState;

BATParticleState bat_io_allocate(void);

void bat_io_free(BATParticleState state);

void bat_io_set_positions(BATParticleState state,
                          void *positions,
                          const uint64_t size,
                          BATDataType type);

void bat_io_set_attribute(BATParticleState state,
                          const char *name,
                          void *data,
                          const uint64_t size,
                          BATDataType type);

// bounds should be [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
void bat_io_set_local_bounds(BATParticleState state, const float *bounds);

void bat_io_set_bytes_per_subfile(BATParticleState state, const uint64_t size);

void bat_io_set_find_best_axis(BATParticleState state, const uint32_t find_best);

void bat_io_set_max_split_imbalance_ratio(BATParticleState state, const float ratio);

void bat_io_set_max_overfull_aggregator_factor(BATParticleState state, const float factor);

void bat_io_set_build_local_trees(BATParticleState state, const uint32_t build_local);

// For testing: disable the adaptive part of the aggregation (defaults to enabled) and
// use a fixed number of aggregators instead
void bat_io_set_fixed_aggregation(BATParticleState state, const uint32_t num_aggregators);

// Write the file out to disk and return the total number of bytes written
uint64_t bat_io_write(BATParticleState state, const char *file_name);

// Collect the JSON formatted performance statistics and return a pointer to the string
// on rank 0. This is a collective call across all ranks. Results are only valid after calling
// bat_io_write
const char *bat_io_get_performance_statistics(BATParticleState state);

#ifdef __cplusplus
}
#endif
