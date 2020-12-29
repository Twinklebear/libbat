#include "profiling.h"
#include <cstring>

using namespace std::chrono;

ProfilingPoint::ProfilingPoint()
{
#ifdef __linux__
    std::memset(&usage, 0, sizeof(rusage));
    getrusage(RUSAGE_SELF, &usage);
#endif
    time = steady_clock::now();
}

float cpu_utilization(const ProfilingPoint &start, const ProfilingPoint &end)
{
#ifdef __linux__
    const double elapsed_cpu =
        end.usage.ru_utime.tv_sec + end.usage.ru_stime.tv_sec -
        (start.usage.ru_utime.tv_sec + start.usage.ru_stime.tv_sec) +
        1e-6f * (end.usage.ru_utime.tv_usec + end.usage.ru_stime.tv_usec -
                 (start.usage.ru_utime.tv_usec + start.usage.ru_stime.tv_usec));

    const double elapsed_wall = duration_cast<duration<double>>(end.time - start.time).count();
    return elapsed_cpu / elapsed_wall * 100.0;
#else
    // TODO: How to do this on windows?
    return -1.f;
#endif
}

size_t elapsed_time_ms(const ProfilingPoint &start, const ProfilingPoint &end)
{
    return duration_cast<milliseconds>(end.time - start.time).count();
}
