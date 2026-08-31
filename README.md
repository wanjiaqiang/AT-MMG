# ATMMG

ATMMG is a C++ prototype for graph-based approximate nearest neighbor
search with trace-derived shortcut edges. The current research branch keeps the
plain L2/u8 graph search path and the shortcut construction path used in local
Fashion-MNIST experiments; older experimental benchmark targets are no longer
part of the build.

## Layout

- `include/ATMMG/ATMMG.hpp` provides the public include entry.
- `include/ATMMG/index/ATMMG_graph/` contains the internal index
  implementation, split into focused implementation fragments under `detail/`.
- `benchmarks/` contains local benchmark drivers for HDF5 datasets and fvecs
  datasets.
- `bin/`, `build*/`, `logs/`, datasets, and temporary experiment outputs are
  ignored by git.

## Build

```powershell
cmake -S . -B build -DATMMG_BUILD_BENCHMARKS=ON
cmake --build build --config Release
```

The HDF5 benchmark is built only when CMake can find the HDF5 C++ package.

## Examples

```powershell
.\bin\Release\ATMMG_hdf5_benchmark.exe sift-128-euclidean.hdf5 1000000 10000
.\bin\Release\ATMMG_fvecs_benchmark.exe base.fvecs query.fvecs 1000000 10000
```

Shortcut-related runtime environment variables use the `ATMMG_` prefix,
for example `ATMMG_TRACE_DEPTH_SHORTCUTS=1`.
