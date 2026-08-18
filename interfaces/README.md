# LibKrylov — C and Fortran interface for Krylov.jl

Exposes the Krylov.jl solvers as a native shared library (`libkrylov`) callable from C, Fortran, and any language with a C FFI.

📖 **Full documentation** (API, examples, return codes, block solvers): <https://jso.dev/Krylov.jl/dev/interfaces/overview/>

Pre-built, self-contained bundles for Linux, macOS and Windows are attached to every [release](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases). The rest of this file is for building the library from source; see the documentation for how to *use* it.

## Requirements

| Tool | Version |
|------|---------|
| Julia | ≥ 1.12 |
| [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) | ≥ 0.3.8 |
| C / Fortran compiler | gcc / clang, gfortran |

[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) wraps Julia's `juliac` compiler and adds `--bundle`, which produces a self-contained library that embeds the Julia runtime (no separate Julia installation required at run time).

## Build

All commands run from the **root of the Krylov.jl repository**.

```bash
# Install JuliaC.jl once (installs juliac into ~/.julia/bin)
julia -e 'import Pkg; Pkg.Apps.add(url="https://github.com/JuliaLang/JuliaC.jl", rev="v0.3.8")'
export PATH="$HOME/.julia/bin:$PATH"

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

# Copy the SuiteSparse libraries into the bundle.
# juliac --bundle does not trace libraries that Julia dlopen's at startup
# (SparseArrays -> SuiteSparse_jll). Without this the bundle runs only on a
# machine that already has Julia. On Windows the libs are in Sys.BINDIR (bin/).
JLIB="$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "..", "lib", "julia"))')"
for name in amd btf camd ccolamd cholmod colamd klu ldl rbio spqr suitesparseconfig umfpack; do
    cp -a "$JLIB"/lib"$name".* interfaces/build/lib/julia/
done
```

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

> **Windows:** use `--output-lib interfaces/build/bin/libkrylov.dll`; the bundle lands in `build/bin/`, and the SuiteSparse libraries are taken from `Sys.BINDIR` (`bin/`).
> **macOS:** replace `.so` with `.dylib`.

Compiling and linking a C or Fortran program against the bundle is documented in the [building guide](https://jso.dev/Krylov.jl/dev/interfaces/building/).

## Directory structure

```
interfaces/
├── src/
│   ├── LibKrylov.jl          # @ccallable functions (compiled by juliac)
│   ├── c_enums.jl            # option-struct mirrors (KrylovOptions, ...)
│   ├── c_operator.jl         # C callback → Julia mul! operator (incl. block)
│   └── c_stores.jl           # AUTO-GENERATED — typed workspace stores
├── scripts/
│   ├── generate_header.jl    # generates include/krylov.h
│   ├── generate_stores.jl    # regenerates src/c_stores.jl (run when adding solvers)
│   ├── solver_table.jl       # single source of truth for the solver list
│   └── trim_sparsearrays.jl  # CI only — strips SparseArrays from the shipped bundle
├── include/
│   ├── krylov.h              # generated C header — do not edit by hand
│   └── krylov.f90            # Fortran bindings (hand-maintained)
├── examples/
│   ├── C/{basic_cg,preconditioning,least_squares,block_gmres}.c
│   └── Fortran/{basic_cg,block_gmres}.f90
├── test/
│   ├── test_libkrylov.jl     # Julia unit tests (no dlopen)
│   ├── C/
│   │   ├── test_all_solvers.c  # convergence of every solver
│   │   ├── test_api.c          # options, preconditioner, warm start, error codes
│   │   └── test_block.c        # block_gmres / block_minres
│   └── Fortran/
│       ├── test_all_solvers.f90
│       └── test_block.f90      # block_gmres / block_minres
└── README.md
```
