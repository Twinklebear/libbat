#pragma once

#include <chrono>
#ifdef __linux__
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#include <unistd.h>
#endif

struct ProfilingPoint {
#ifdef __linux__
    rusage usage;
#endif
    std::chrono::steady_clock::time_point time;

    ProfilingPoint();
};

float cpu_utilization(const ProfilingPoint &start, const ProfilingPoint &end);

size_t elapsed_time_ms(const ProfilingPoint &start, const ProfilingPoint &end);
