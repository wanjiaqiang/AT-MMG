#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DATMMG_BUILD_BENCHMARKS=ON
cmake --build build --config Release

# HDF5 benchmark, for ann-benchmarks style datasets.
./bin/Release/ATMMG_hdf5_benchmark ./data/sift-128-euclidean.hdf5 1000000 10000

# FVECS benchmark, for raw vector files.
./bin/Release/ATMMG_fvecs_benchmark ./data/base.fvecs ./data/query.fvecs 1000000 10000
