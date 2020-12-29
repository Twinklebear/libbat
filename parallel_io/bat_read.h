#pragma once

#include <stdint.h>
#include "bat_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *BATParticleFile;

BATParticleFile bat_io_open(const char *file_name);

void bat_io_close(BATParticleFile file);

// Query the global bounds of the data set
const float *bat_io_get_global_bounds(BATParticleFile file);

uint64_t bat_io_get_num_attributes(BATParticleFile file);

uint64_t bat_io_get_num_points(BATParticleFile file);

// Query the description of one of the attributes (name, value range, data type)
void bat_io_get_attribute_description(
    BATParticleFile file, uint64_t id, const char **name, float *range, BATDataType *type);

// bounds should be [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
void bat_io_set_read_bounds(BATParticleFile file, const float *bounds);

// Read the file, the loaded particles can then be fetched from the file state
// returns the number of particles read on this rank
// TODO/NOTE: This does a checkpoint-restart style read only right now
uint64_t bat_io_read(BATParticleFile file);

// Get the pointer to the positions data read from the file
void bat_io_get_positions(BATParticleFile file,
                          const void **positions,
                          uint64_t *size,
                          BATDataType *type);

// Get the pointer to the attribute data read from the file
void bat_io_get_attribute(BATParticleFile file,
                          const char *name,
                          const void **data,
                          uint64_t *size,
                          BATDataType *type);

#ifdef __cplusplus
}
#endif
