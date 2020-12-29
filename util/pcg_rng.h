#pragma once

#include <cmath>

// http://www.pcg-random.org/download.html
struct PCGRand {
    uint64_t state = 0;
    // Just use stream 1

    PCGRand() = default;
    inline PCGRand(uint32_t seed)
    {
        random();
        state += seed;
        random();
    }

    inline uint32_t random()
    {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + 1;
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    inline float randomf()
    {
        return std::ldexp(static_cast<double>(random()), -32);
    }
};

