# AT-MMG

AT-MMG (Axis-Tree-Guided Multi-Entry Monotonic Graph) is a two-stage
approximate nearest neighbor search framework. It first uses an axis tree to
anchor a query to a local partition, and then performs high-recall graph
search from the corresponding entry set in a multi-entry monotonic graph. The
design couples tree-based localization with graph-based refinement to shorten
search paths while preserving accuracy.

## Layout

- `include/ATMMG/ATMMG.hpp` provides the public include entry.
- `include/ATMMG/index/ATMMG_graph/` contains the internal index
  implementation, split into focused implementation fragments under `detail/`.
- `benchmarks/` contains benchmark drivers for HDF5 datasets and fvecs
  datasets.

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

AT-MMG runtime environment variables use the `ATMMG_` prefix.
