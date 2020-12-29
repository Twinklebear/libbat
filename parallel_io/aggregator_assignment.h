#pragma once

#include <vector>
#include "aggregation_tree.h"
#include "particle_data.h"

struct AggregatorInfo {
    int aggregator_rank = -1;
    std::vector<int> clients;
    std::vector<int> client_particle_counts;
    // NOTE: this will only have data on rank 0, later we may want it also
    // on the aggregators for in situ stuff
    AggregationTree tree;

    AggregatorInfo(const int rank);

    AggregatorInfo() = default;
};

AggregatorInfo compute_aggregator_assignment(ParticleData *data);
