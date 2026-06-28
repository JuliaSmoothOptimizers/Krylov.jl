# LibKrylov — C and Fortran interface for Krylov.jl

Exposes the Krylov.jl solvers as a native shared library (`libkrylov.so`) callable from C, Fortran, and any language with a C FFI.

## Requirements

| Tool | Version |
|------|---------|
| Julia | ≥ 1.12 |
| [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) | latest |
| C compiler | gcc / clang |

[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) wraps Julia's `juliac` compiler and adds `--bundle`, which produces a **self-contained** library that embeds the Julia runtime — no separate Julia installation required at runtime.

## Build

All commands run from the **root of the Krylov.jl repository**.

### With JuliaC.jl (recommended — self-contained bundle)

```bash
# Install JuliaC.jl once (Julia app — installs juliac into ~/.julia/bin)
julia -e 'import Pkg; Pkg.Apps.add(url="https://github.com/JuliaLang/JuliaC.jl", rev="v0.3.2")'
export PATH="$HOME/.julia/bin:$PATH"   # add to ~/.bashrc to make permanent

# Build the bundle (library + embedded Julia runtime)
juliac \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/build \
    --output-lib interfaces/build/lib/libkrylov.so \
    interfaces/src/LibKrylov.jl

# Generate the C header
julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
cp interfaces/include/krylov.h   interfaces/build/include/
cp interfaces/include/krylov.f90 interfaces/build/include/

# Compile a C example
gcc -o basic_cg interfaces/examples/C/basic_cg.c \
    -I interfaces/build/include \
    interfaces/build/lib/libkrylov.so \
    -Wl,-rpath,'$ORIGIN/../lib/julia'
```

The `--bundle` flag produces a **relocatable** directory:

```
interfaces/build/
├── lib/
│   ├── libkrylov.so     ← the library
│   └── julia/           ← embedded Julia runtime (no system Julia needed)
└── include/
    ├── krylov.h
    └── krylov.f90
```

> **Windows:** use `--output-lib interfaces/build/bin/libkrylov.dll`; the bundle lands in `build/bin/`.  
> **macOS:** replace `.so` with `.dylib` and use `-Wl,-rpath,@loader_path/../lib/julia`.

**Output sizes** (Linux x86-64, all solvers × 4 precisions):

| Build | Size |
|-------|------|
| No trim | ~269 MB |
| `--trim=safe` | ~19 MB |

## Run the examples

```bash
./build/basic_cg
# Solved: yes   niter: 3   time: 3.2e-05 s
# x = [ 1.00 1.00 1.00 1.00 1.00 ]
```

## API overview

```c
#include "krylov.h"

/* Callback type: computes y = A*x  or  y = A'*x  or  y = M\x */
typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

/* 1. Create a workspace for a named solver */
KrylovWorkspaceOptions krylov_default_workspace_options(void);  /* memory / window  */

int krylov_workspace_create(KrylovSolverType solver, /* KRYLOV_CG, KRYLOV_GMRES, ...  */
                            int m, int n,             /* operator dimensions           */
                            KrylovDataType dtype,     /* KRYLOV_FLOAT64, ...           */
                            KrylovDeviceType device,  /* KRYLOV_CPU                    */
                            const KrylovWorkspaceOptions *wopts, /* NULL = defaults    */
                            void **ws_out);           /* receives the handle           */

/* 2. Solve */
KrylovOptions krylov_default_options(void);  /* atol/rtol/itmax/verbose/lambda/tau/nu */

int krylov_solve(void *ws,
                 KrylovMatvec matvec_A,   /* y = A*x  (required)               */
                 KrylovMatvec matvec_At,  /* y = A'*x (NULL if not needed)     */
                 KrylovMatvec matvec_M,   /* y = M\x  (NULL = no precond.)     */
                 const void *b,           /* right-hand side (size m)          */
                 const void *c,           /* second RHS (NULL if not needed)   */
                 void *userdata,          /* forwarded to every callback        */
                 const KrylovOptions *opts); /* NULL = solver defaults          */

/* 3. Retrieve results */
int    krylov_get_x(void *ws, void *x, int n);   /* primal solution             */
int    krylov_get_y(void *ws, void *y, int m);   /* dual solution (TriCG, ...)  */
int    krylov_is_solved(void *ws);                /* 1=yes, 0=no, -1=error      */
int    krylov_niter(void *ws);
double krylov_elapsed_time(void *ws);             /* seconds                    */

/* 4. Optional: warm start */
int krylov_warm_start(void *ws, const void *x0, int n);

/* 5. Free */
int krylov_workspace_free(void *ws);
```

### Enumerators and option structs

```c
typedef enum { KRYLOV_FLOAT32=0, KRYLOV_FLOAT64=1,
               KRYLOV_COMPLEX32=2, KRYLOV_COMPLEX64=3 } KrylovDataType;

typedef enum { KRYLOV_CPU=0 } KrylovDeviceType;

/* Construction-time options (krylov_workspace_create). 0 = solver default. */
typedef struct {
    int memory;  /* GMRES / FGMRES / FOM / DIOM / DQGMRES / GPMR  (default 20) */
    int window;  /* MINRES / SYMMLQ / LSQR / LSMR / LSLQ          (default 5)  */
} KrylovWorkspaceOptions;

/* Solve-time options (krylov_solve). NaN/0 = solver default. */
typedef struct {
    double atol, rtol;       /* tolerances                                    */
    int    itmax, verbose;   /* max iterations / verbosity                    */
    double lambda;           /* regularisation (LSQR / LSMR / CGLS / ...)     */
    double tau, nu;          /* TriCG / TriMR diagonal parameters             */
} KrylovOptions;
```

### Which solvers need `matvec_At`?

| Pass `NULL` | Pass a callback |
|-------------|----------------|
| CG, CR, MINRES, MINRES-QLP, SYMMLQ, GMRES, FGMRES, FOM, DIOM, DQGMRES, BiCGSTAB, CGS, CAR, MINARES, TriCG, TriMR, GPMR | BiLQ, QMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR, LSLQ, LSQR, LSMR, CGLS, CRLS, CGNE, CRMR, CRAIG, CRAIGMR, LNLQ |

### Minimal example (CG, double precision)

```c
#include "krylov.h"

static void my_matvec(const void *x, void *y, void *data) {
    /* fill y = A*x */
}

int main(void) {
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

    KrylovOptions opts = krylov_default_options();
    opts.atol = 1e-10;
    opts.rtol = 1e-10;
    krylov_solve(ws, my_matvec, NULL, NULL, b, NULL, NULL, &opts);
    krylov_get_x(ws, x, n);
    krylov_workspace_free(ws);
}
```

### Block solvers (multiple right-hand sides)

`block_gmres` and `block_minres` solve `A X = B` for an `m×p` block `B` at once.
They have a parallel API — `krylov_block_workspace_create`, `krylov_block_solve`,
`krylov_block_get_X`, … — with a block matvec that also receives the block width:

```c
typedef enum { KRYLOV_BLOCK_GMRES = 0, KRYLOV_BLOCK_MINRES = 1 } KrylovBlockSolverType;

/* Y = A*X for a block of p columns; X is n*p, Y is m*p, both column-major */
typedef void (*KrylovBlockMatvec)(const void *X, void *Y, int p, void *userdata);

int krylov_block_workspace_create(KrylovBlockSolverType solver, int m, int n, int p,
                                  KrylovDataType dtype, KrylovDeviceType device,
                                  const KrylovWorkspaceOptions *wopts, void **ws_out);
int krylov_block_solve(void *ws, KrylovBlockMatvec matvec_A, KrylovBlockMatvec matvec_M,
                       const void *B, void *userdata, const KrylovOptions *opts);
int krylov_block_get_X(void *ws, void *X, int n, int p);
/* + krylov_block_{is_solved,niter,elapsed_time,warm_start,workspace_free} */
```

Blocks are column-major and `B` must have full column rank. See the
[documentation](https://jso.dev/Krylov.jl/dev/interfaces/overview/) for details.

## Directory structure

```
interfaces/
├── src/
│   ├── LibKrylov.jl          # @ccallable functions (compiled by juliac)
│   ├── c_enums.jl            # KrylovDataType / KrylovDeviceType enum helpers
│   ├── c_operator.jl         # COperator: C callback → Julia mul! operator
│   └── c_stores.jl           # AUTO-GENERATED — typed workspace stores
├── scripts/
│   ├── generate_header.jl    # generates include/krylov.h
│   ├── generate_stores.jl    # regenerates src/c_stores.jl (run when adding solvers)
│   └── solver_table.jl       # single source of truth for solver list
├── include/
│   ├── krylov.h              # generated C header — do not edit by hand
│   └── krylov.f90            # Fortran bindings
├── examples/
│   ├── C/
│   │   └── basic_cg.c        # CG on tridiag(-1,2,-1)
│   └── Fortran/
│       └── basic_cg.f90
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
