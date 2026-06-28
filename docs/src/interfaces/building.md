# [Building libkrylov](@id building-libkrylov)

Pre-built, self-contained bundles for Linux (x86-64, aarch64), macOS (arm64, x86-64) and Windows (x86-64) are attached to every [release](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases) starting `v0.10.7`.
This page describes how to build the library from source and how to compile and link a C or Fortran program against it.

## Requirements

| Tool | Version |
|------|---------|
| Julia | ≥ 1.12 |
| JuliaC.jl | ≥ 0.3.8 |
| C / Fortran compiler | gcc / clang, gfortran |

[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) wraps Julia's `juliac` compiler and adds `--bundle`, which produces a self-contained library that embeds the Julia runtime, so no separate Julia installation is required at run time.

## Build from source

All commands run from the root of the Krylov.jl repository.

```bash
# Install JuliaC.jl once (it installs juliac into ~/.julia/bin)
julia -e 'import Pkg; Pkg.Apps.add(url="https://github.com/JuliaLang/JuliaC.jl", rev="v0.3.8")'
export PATH="$HOME/.julia/bin:$PATH"   # add to ~/.bashrc to make it permanent

# Build the bundle (library + embedded Julia runtime)
juliac \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/build \
    --output-lib interfaces/build/lib/libkrylov.so \
    interfaces/src/LibKrylov.jl

# Generate the headers and copy them next to the library
julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
cp interfaces/include/krylov.h   interfaces/build/include/
cp interfaces/include/krylov.f90 interfaces/build/include/

# Copy the SuiteSparse libraries into the bundle (see the note below)
JLIB="$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "..", "lib", "julia"))')"
for name in amd btf camd ccolamd cholmod colamd klu ldl rbio spqr suitesparseconfig umfpack; do
  cp -a "$JLIB"/lib"$name".* interfaces/build/lib/julia/
done
```

!!! warning "SuiteSparse must be copied manually"
    `juliac --bundle` only copies the libraries it can trace statically. The
    SuiteSparse stack (`libbtf`, `libcholmod`, `libumfpack`, ...) is loaded
    dynamically by Julia at startup (`SparseArrays` is a dependency of Krylov),
    so `juliac` does not see it. Without the copy step above, the bundle runs
    fine on a machine that has Julia installed but fails on a clean machine with
    `could not load library "libbtf.so.2"`. On Windows the libraries live in
    `Sys.BINDIR` (the `bin/` folder) instead of `lib/julia`.

The `--bundle` flag produces a relocatable directory:

```
interfaces/build/
├── lib/
│   ├── libkrylov.so     ← the library
│   └── julia/           ← embedded Julia runtime (no system Julia needed)
└── include/
    ├── krylov.h
    └── krylov.f90
```

The `--trim=safe` flag is what keeps the bundle small by discarding unreachable code:

| Build | Size (Linux x86-64, all solvers × 4 precisions) |
|-------|------|
| No trim | ~269 MB |
| `--trim=safe` | ~19 MB |

!!! note "Platform differences"
    - **Windows:** use `--output-lib interfaces/build/bin/libkrylov.dll`; the bundle lands in `build/bin/`.
    - **macOS:** replace `.so` with `.dylib`.

## Compiling and linking a C program

Link against the bundled `libkrylov`, and point the runtime loader at the embedded Julia runtime with an `-rpath`:

```bash
gcc -O2 -o basic_cg interfaces/examples/C/basic_cg.c \
    -I interfaces/build/include \
    interfaces/build/lib/libkrylov.so \
    -Wl,-rpath,'$ORIGIN/../lib/julia'
```

On macOS, use `-Wl,-rpath,@loader_path/../lib/julia` instead. Add `-lm` if your program uses the math library.

## Compiling and linking a Fortran program

```bash
gfortran -O2 -o basic_cg interfaces/examples/Fortran/basic_cg.f90 \
    interfaces/build/lib/libkrylov.so
```

The `include 'krylov.f90'` line resolves relative to the source file, so either keep `krylov.f90` next to your program or pass `-I interfaces/build/include` and adjust the `include` path accordingly.

## Running

```bash
./basic_cg
# Solved: yes   niter: 3   time: 3.2e-05 s
# x = [ 1.00 1.00 1.00 1.00 1.00 ]
```

If the loader cannot find `libkrylov` or the embedded Julia runtime at run time, add the bundle directories to the library search path, for example on Linux:

```bash
export LD_LIBRARY_PATH="$PWD/interfaces/build/lib:$PWD/interfaces/build/lib/julia:$LD_LIBRARY_PATH"
```

(use `DYLD_LIBRARY_PATH` on macOS and `PATH` on Windows).
