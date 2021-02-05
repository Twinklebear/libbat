# libbat

This repository contains the code implementing the 
the adaptive, spatially aware two-phase I/O strategy
for particle data and the low overhead multiresolution data layout
described in our IPDPS 2021 paper:
[*Adaptive Spatially Aware I/O for Multiresoluton Particle Data Layouts*](https://www.willusher.io/publications/aggtree-io).
The scripts used to run the benchmarks are available [here](https://github.com/Twinklebear/libbat-benchmark-scripts).

## Dependencies

- [TBB](https://github.com/oneapi-src/oneTBB)
- [GLM](https://github.com/g-truc/glm), 0.9.9.8 or higher

Optional for viewer:

- [SDL2](https://www.libsdl.org/)


# Reproducibility Information for IPDPS 2021

The parallel I/O benchmarks were run on Stampede2 using the
Skylake Xeon nodes and Summit. On Stampede2 data was written
to `/scratch` with a stripe size of 8MB and stripe count of 32.
On Summit data was written to `/gpfs`. Runs on both machines
are done using a process per-core. The benchmark scripts used
for our library and IOR are available in the [scripts repository](https://github.com/Twinklebear/libbat-benchmark-scripts).
The modified version of ExaMPM's mini-app used to generate the
dam break data sets is available [here](https://github.com/Twinklebear/ExaMPM-libbat).

On Stampede2 the following modules were loaded:
```
git/2.24.1
cmake/3.16.1
TACC
qt5/5.11.2
python3/3.6.1
phdf5/1.10.4
autotools/1.1
xalt/2.8
libfabric/1.7.0
gcc/7.1.0
impi/18.0.2
```

On Summit the following modules were loaded:
```
hsi/5.0.2.p5
lsftools/2.0
DefApps
cmake/3.15.2
hdf5/1.10.4
git/2.20.1
xalt/1.2.0
darshan-runtime/3.1.7
gcc/9.1.0
spectrum-mpi/10.3.1.2-20200121
python/3.7.0-anaconda3-5.3.0
```

For our IOR comparisons we built the latest IOR from [Github](https://github.com/hpc/ior),
at the time this was commit hash `3562a35`.
When building our library we use TBB version 2020u1 and [glm](https://github.com/g-truc/glm)
at commit hash `efbfecab` (now released as 0.9.9.8).

The visualization read benchmarks were performed on a desktop
with an i9-9920X CPU and 128GB of RAM running Ubuntu 19.04
(GNU/Linux 5.0.0-38-generic x86_64). The data was read from a
1TB Samsung 860 NVMe drive. The compiler used was gcc 8.3.0

