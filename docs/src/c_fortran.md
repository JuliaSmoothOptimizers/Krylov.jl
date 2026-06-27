# C and Fortran interfaces

!!! warning "Prototype"
    This interface is still in early development. Not all Krylov.jl features are exposed yet (e.g. shifts, GPU devices). The API may change in future releases.

Krylov.jl ships `libkrylov`, a native shared library that exposes its solvers to C, Fortran, Python, R, or any other language that can call C code.

Pre-built artifacts for Linux (x86-64, aarch64), macOS (arm64, x86-64) and Windows (x86-64) are available on the [Releases](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases) page.

## How it works

`libkrylov` is the whole of Krylov.jl compiled ahead of time into a shared
library, with the Julia runtime **embedded** in the bundle.
From the caller's point of view it is an ordinary C library: there is no Julia process to start and no `.jl` file to ship.
It is only `libkrylov.so` (or `.dylib` / `.dll`) and the two headers `krylov.h` / `krylov.f90`.

The design rests on three ideas:

- **Opaque workspace handle.** `krylov_workspace_create` allocates a *typed*
  Krylov workspace (it holds all the internal vectors a solver needs) and hands
  back a `void *`. You never dereference it. You pass it to the other functions.
  The library keeps the workspace alive (it is rooted against the Julia garbage
  collector) until you call `krylov_workspace_free`.

- **Matrix-free by callbacks.** Krylov methods only ever touch the operator
  through products `A * x` (and possibly `Aᴴ * x` or `M⁻¹ * x`). You never pass a
  matrix; you pass C function pointers that compute those products. This means
  the same interface drives a dense matrix, a sparse matrix, a stencil, or a
  fully matrix-free operator, in any language with a C ABI.

- **Concrete types, no dynamic dispatch.** A workspace is
  created for one `(solver, precision)` pair, so every product, dot, and axpy
  runs through statically-typed, fully specialised Julia code.

The lifecycle is always the same, regardless of the solver:

1. **Create** a workspace with `krylov_workspace_create`
2. **Solve** with `krylov_solve`, passing matrix-vector product callbacks
3. **Retrieve** the solution with `krylov_get_x` (and `krylov_get_y` for two-solution solvers)
4. **Free** the workspace with `krylov_workspace_free`

A single workspace can be reused for several `krylov_solve` calls (e.g. with a
different right-hand side, or after `krylov_warm_start`); only `create`/`free`
allocate and release memory.

## API overview

```c
// Callback signature: computes y = A * x, y = Aᴴ * x, or y = M \ x
typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

// Construction-time options (memory / window); NULL = solver defaults
KrylovWorkspaceOptions krylov_default_workspace_options(void);

int krylov_workspace_create(KrylovSolverType solver,   // KRYLOV_CG, KRYLOV_GMRES, ...
                            int m, int n,
                            KrylovDataType precision,  // KRYLOV_FLOAT64, ...
                            KrylovDeviceType device,   // KRYLOV_CPU
                            const KrylovWorkspaceOptions *wopts, // NULL = defaults
                            void **ws_out);

// Solve-time options (atol, rtol, itmax, verbose, lambda, tau, nu); NULL = defaults
KrylovOptions krylov_default_options(void);

int krylov_solve(void *ws,
                 KrylovMatvec matvec_A,   // required
                 KrylovMatvec matvec_At,  // NULL if not needed
                 KrylovMatvec matvec_M,   // NULL = no preconditioner
                 const void *b,           // right-hand side
                 const void *c,           // second RHS, NULL if not needed
                 void *userdata,
                 const KrylovOptions *opts); // NULL = solver defaults

int    krylov_get_x(void *ws, void *x, int n);
int    krylov_get_y(void *ws, void *y, int m);
int    krylov_is_solved(void *ws);         // 1 = converged
int    krylov_niter(void *ws);
double krylov_elapsed_time(void *ws);
int    krylov_warm_start(void *ws, const void *x0, int n);
int    krylov_workspace_free(void *ws);
```

## The matvec callbacks

A callback receives three raw pointers and returns nothing:

```c
void matvec(const void *x, void *y, void *userdata);
```

- `x` is the **input** vector (read-only); `y` is the **output** vector you must
  fill; `userdata` is the opaque pointer you handed to `krylov_solve`, forwarded
  unchanged to every callback. Use it to carry whatever your product needs (the
  matrix, problem sizes, scratch buffers).
- Both vectors point to arrays of the element type you chose at creation
  (`double`, `float`, `double _Complex`, `float _Complex`). Cast them
  accordingly before use.
- The library owns `x` and `y`; do **not** keep, free, or resize them. They are
  only valid for the duration of the call.

The three slots play different roles, and their vector lengths follow the
operator shape (`m` rows, `n` columns):

| Slot        | Computes      | `x` length | `y` length | When to pass |
|-------------|---------------|:----------:|:----------:|--------------|
| `matvec_A`  | `y = A * x`   | `n`        | `m`        | always (required) |
| `matvec_At` | `y = Aᴴ * x`  | `m`        | `n`        | solvers that use the adjoint (see below) |
| `matvec_M`  | `y = M⁻¹ * x` | `n`        | `n`        | only if preconditioning |

For square systems `m == n`, so all three lengths coincide, that is the common
case (like CG, GMRES).
The distinction matters for least-squares / least-norm solvers where `A` is rectangular.

!!! note "GPMR is special"
    For `KRYLOV_GPMR` the second callback slot is **not** the adjoint of `A`: it
    is a separate operator `B` (size `n × m`) used by the block formulation
    `[0 A; B 0]`. Pass `B`'s product where `matvec_At` would normally go.

## Data types and precision

A workspace is created for exactly one element type, selected with
`KrylovDataType`:

| Enum               | Julia type   | C element type    |
|--------------------|--------------|-------------------|
| `KRYLOV_FLOAT32`   | `Float32`    | `float`           |
| `KRYLOV_FLOAT64`   | `Float64`    | `double`          |
| `KRYLOV_COMPLEX32` | `ComplexF32` | `float _Complex`  |
| `KRYLOV_COMPLEX64` | `ComplexF64` | `double _Complex` |

Every buffer that crosses the boundary — `b`, `c`, the callback's `x` / `y`,
`x0`, and the destinations of `krylov_get_x` / `krylov_get_y`, must be an array
of that element type.
The complex layout is the standard interleaved real/imaginary one, identical to C99 `_Complex` and to Julia's `Complex{T}`, so
no conversion is needed.
The only device currently available is `KRYLOV_CPU = 0`.

## Choosing a solver

`KrylovSolverType` lists every exposed solver (`KRYLOV_CG`, `KRYLOV_GMRES`, ...).
Three properties decide how you call `krylov_solve`:

**Does it need `matvec_At`?** Pass `NULL` for the left column, a real callback
for the right one:

| Pass `NULL` for `matvec_At` | Provide `matvec_At` |
|-----------------------------|---------------------|
| CG, CR, CAR, MINRES, MINRES-QLP, MINARES, SYMMLQ, GMRES, FGMRES, FOM, DIOM, DQGMRES, BiCGSTAB, CGS, TriCG, TriMR | BiLQ, QMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR, LSLQ, LSQR, LSMR, CGLS, CRLS, CGNE, CRMR, CRAIG, CRAIGMR, LNLQ, GPMR\* |

\* GPMR uses that slot for `B`, not `Aᴴ` (see the note above).

**Does it take a second right-hand side `c`?** Required by TriCG, TriMR, BiLQR,
TriLQR, USYMLQ, USYMQR, USYMLQR and GPMR; pass `NULL` for `c` otherwise.

**Does it produce a second solution `y`?** Retrieve it with `krylov_get_y`
(length `m`) for TriCG, TriMR, GPMR, BiLQR and TriLQR. For every other solver
`krylov_get_y` returns `-2`.

## Options

Options are split by *when* they are consumed:

| Struct | Passed to | Fields |
|--------|-----------|--------|
| `KrylovWorkspaceOptions` | `krylov_workspace_create` | `memory`, `window` |
| `KrylovOptions` | `krylov_solve` | `atol`, `rtol`, `itmax`, `verbose`, `lambda`, `tau`, `nu` |

Always initialise a struct from its `krylov_default_*` helper, then override only
the fields you need; every field has a sentinel (`0` for ints, `NaN` for doubles)
meaning *use the solver default*. Fields irrelevant to a given solver are
silently ignored, so it is safe to set, say, `lambda` even when running CG.

**Construction-time** (`KrylovWorkspaceOptions`) — these size internal storage,
so they belong to creation rather than to the solve:

- `memory` — default `20`. Its meaning depends on the solver:
  - **DIOM, DQGMRES** are genuinely limited-memory: `memory` is the number of
    recent basis vectors a new vector is orthogonalized against (the truncation
    parameter), so it directly changes the algorithm.
  - **GMRES, FGMRES, FOM, GPMR** are full methods here: `memory` only sizes the
    initial storage and the basis grows on demand, so it is a performance hint,
    not an algorithmic knob. (The restarted GMRES(k) variant is not exposed yet.)
- `window` — number of past residuals used by the residual-norm estimate of
  MINRES, SYMMLQ, LSQR, LSMR and LSLQ; default `5`.

**Solve-time** (`KrylovOptions`):

- `atol`, `rtol` — absolute / relative stopping tolerances. `NaN` → `√eps(T)`
  for the chosen precision.
- `itmax` — iteration cap. `0` → the solver default (a small multiple of the
  problem size).
- `verbose` — `0` is silent; a positive value prints convergence information
  every `verbose` iterations.
- `lambda` — Tikhonov regularisation for the least-squares / least-norm solvers
  (LSQR, LSMR, CGLS, CRLS, LNLQ, LSLQ, CRAIG, CRAIGMR); `0.0` = none.
- `tau`, `nu` — diagonal scalings of the saddle-point system solved by TriCG and
  TriMR; `NaN` → `1.0` and `-1.0` respectively.

```c
// DQGMRES truncated to the 10 most recent Krylov basis vectors
KrylovWorkspaceOptions wopts = krylov_default_workspace_options();
wopts.memory = 10;
krylov_workspace_create(KRYLOV_DQGMRES, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &wopts, &ws);
```

## Return codes

Every function returns an `int` status (`krylov_elapsed_time` returns a
`double`). The conventions are:

| Function | Success | Other values |
|----------|:-------:|--------------|
| `krylov_workspace_create` | `0` | `-1` internal error · `-2` unknown `(solver, precision)` combination |
| `krylov_solve` | `0` | `-1` internal error (e.g. a callback threw) |
| `krylov_get_x` | `0` | `-1` error (bad handle / size) |
| `krylov_get_y` | `0` | `-1` error · `-2` solver has a single solution |
| `krylov_warm_start` | `0` | `-1` error · `-2` solver does not support warm start |
| `krylov_workspace_free` | `0` | `1` handle not found |
| `krylov_is_solved` | `1` converged | `0` not converged · `-1` error |
| `krylov_niter` | iteration count | `-1` error |
| `krylov_elapsed_time` | seconds (`double`) | `-1.0` error |

Always check the status of `krylov_workspace_create` and `krylov_solve` before
reading results, and use `krylov_is_solved` to confirm convergence — a non-zero
`krylov_solve` means the run failed, whereas `krylov_is_solved == 0` means it ran
but did not reach the tolerance within `itmax`.

## Workspace lifetime

`krylov_workspace_create` is the only allocating call; `krylov_workspace_free`
is the only releasing one. Between them you may call `krylov_solve` (and the
accessors) as many times as you like — for example to solve `A x = b₁`, then
`A x = b₂` with the same operator, reusing all the internal vectors. The handle
becomes invalid the moment you free it; do not touch it afterwards. Each created
workspace must be freed exactly once to avoid leaking the (potentially large)
internal storage.

## Warm start

To seed a solve with an initial guess `x0`, call `krylov_warm_start(ws, x0, n)`
**before** `krylov_solve`. The guess is copied into the workspace and consumed
by the next solve. Solvers that do not support warm starting return `-2`. See
[Warm-start](@ref warm-start) for the underlying mechanism.

## Preconditioning

Pass a `matvec_M` that applies `M⁻¹` (i.e. computes `y = M \ x`, the *action* of
the preconditioner) to use `M` as a left preconditioner; pass `NULL` for an
unpreconditioned solve. The callback sees vectors of length `n` and, like the
operator, is matrix-free — `M` can be an incomplete factorization, a sparse
approximate inverse, a multigrid V-cycle, or anything else you can evaluate.
See [Preconditioners](@ref preconditioners) for guidance on choosing one.

## C example

Solve a 5×5 tridiagonal SPD system with CG in double precision
([`examples/C/basic_cg.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/C/basic_cg.c)):

```c
#include <stdio.h>
#include "krylov.h"

#define N 5

typedef struct { int n; double diag[N]; double off[N-1]; } TriDiag;

static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x  = (const double *)xv;
  double       *y  = (double *)yv;
  const TriDiag *A = (const TriDiag *)userdata;
  for (int i = 0; i < A->n; i++) {
    y[i] = A->diag[i] * x[i];
    if (i > 0)     y[i] += A->off[i-1] * x[i-1];
    if (i < A->n-1) y[i] += A->off[i]   * x[i+1];
  }
}

int main(void)
{
  TriDiag A = { .n = N };
  for (int i = 0; i < N;   i++) A.diag[i] =  2.0;
  for (int i = 0; i < N-1; i++) A.off[i]  = -1.0;

  double b[N] = {1.0, 0.0, 0.0, 0.0, 1.0};
  double x[N];

  void *ws = NULL;
  krylov_workspace_create(KRYLOV_CG, N, N, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10;
  opts.rtol = 1e-10;
  krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, &opts);
  krylov_get_x(ws, x, N);

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  for (int i = 0; i < N; i++) printf(" %.2f", x[i]);
  printf("\n");

  krylov_workspace_free(ws);
  return 0;
}
```

More complete C programs are kept in the repository and exercised in CI:

- [`test/C/test_all_solvers.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_all_solvers.c) — every solver, on SPD / non-symmetric / least-squares problems
- [`test/C/test_api.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_api.c) — options, preconditioner, warm start, workspace reuse, error codes, `Float32`
- [`test/C/test_block.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_block.c) — `block_gmres` / `block_minres`

## Fortran example

The same problem using the Fortran interfaces (`krylov.f90` included in the bundle),
[`examples/Fortran/basic_cg.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/Fortran/basic_cg.f90):

```fortran
program basic_cg
  use iso_c_binding
  implicit none
  include 'krylov.f90'

  integer, parameter    :: n = 5
  real(c_double), target :: diag(n), off(n-1), b(n), x(n)
  type(KrylovOptions), target :: opts
  type(c_ptr)           :: ws
  integer(c_int)        :: ret

  diag = 2.0_c_double ; off = -1.0_c_double
  b = 0.0_c_double ; b(1) = 1.0_c_double ; b(n) = 1.0_c_double

  ret = krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &
                                c_null_ptr, ws)

  opts = krylov_default_options()
  opts%atol = 1d-10 ; opts%rtol = 1d-10
  ret = krylov_solve(ws, c_funloc(matvec_A), c_null_funptr, c_null_funptr, &
                     c_loc(b), c_null_ptr, c_loc(diag), c_loc(opts))
  ret = krylov_get_x(ws, c_loc(x), int(n, c_int))

  write(*,*) "Solved:", krylov_is_solved(ws) == 1, "  niter:", krylov_niter(ws)
  write(*,'(5F6.2)') x

  ret = krylov_workspace_free(ws)

contains
  subroutine matvec_A(x_ptr, y_ptr, userdata) bind(c)
    type(c_ptr), value :: x_ptr, y_ptr, userdata
    real(c_double), pointer :: xv(:), yv(:), dg(:)
    integer :: i
    call c_f_pointer(x_ptr, xv, [n]) ; call c_f_pointer(y_ptr, yv, [n])
    call c_f_pointer(userdata, dg, [n])
    do i = 1, n
      yv(i) = dg(i)*xv(i)
      if (i > 1) yv(i) = yv(i) - xv(i-1)
      if (i < n) yv(i) = yv(i) - xv(i+1)
    end do
  end subroutine
end program
```

### Fortran specifics

A few rules make the binding work:

- `include 'krylov.f90'` goes **after** `implicit none`. It declares the enum
  parameters, the `KrylovOptions` / `KrylovWorkspaceOptions` derived types (both
  `bind(c)`), and the `interface` blocks for every function.
- Anything whose address you take with `c_loc` must have the `target` attribute —
  including the option structs (`type(KrylovOptions), target :: opts`).
- Vectors and structs are passed as `c_loc(array)` / `c_loc(opts)`; pass
  `c_null_ptr` for an absent `b`/`c`/`opts`/`wopts`.
- Callbacks are passed as `c_funloc(my_sub)` and must be `bind(c)` subroutines
  with three `type(c_ptr), value` arguments; pass `c_null_funptr` for an unused
  slot. Inside, recover Fortran arrays with `c_f_pointer`.

A program covering every solver is available at
[`test/Fortran/test_all_solvers.f90`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/Fortran/test_all_solvers.f90).

## Block Krylov solvers

`block_gmres` and `block_minres` solve a system with several right-hand sides at
once, `A X = B`, where `B` and `X` are `m×p` / `n×p` blocks. They have their own
parallel API, mirroring the single-vector one:

```c
typedef enum { KRYLOV_BLOCK_GMRES = 0, KRYLOV_BLOCK_MINRES = 1 } KrylovBlockSolverType;

// Block matvec: computes Y = A*X (or Y = M\X) for a block of p columns.
//   X : input  block (n*p, column-major)
//   Y : output block (m*p, column-major)
//   p : number of columns
typedef void (*KrylovBlockMatvec)(const void *X, void *Y, int p, void *userdata);

int    krylov_block_workspace_create(KrylovBlockSolverType solver, int m, int n, int p,
                                     KrylovDataType dtype, KrylovDeviceType device,
                                     const KrylovWorkspaceOptions *wopts, void **ws_out);
int    krylov_block_solve(void *ws, KrylovBlockMatvec matvec_A,
                          KrylovBlockMatvec matvec_M,  // NULL = no preconditioner
                          const void *B, void *userdata, const KrylovOptions *opts);
int    krylov_block_get_X(void *ws, void *X, int n, int p);
int    krylov_block_is_solved(void *ws);
int    krylov_block_niter(void *ws);
double krylov_block_elapsed_time(void *ws);
int    krylov_block_warm_start(void *ws, const void *X0, int n, int p);
int    krylov_block_workspace_free(void *ws);
```

Key differences from the single-vector API:

- **Blocks are column-major**, exactly like a Fortran/LAPACK matrix: entry `(i, j)`
  of an `n×p` block is at offset `i + j*n`. `B`, `X`, `X0`, and the callback's
  `X` / `Y` all use this layout.
- **The callback receives the block width `p`** so it can apply `A` to all
  columns in one call (`Y = A·X`).
- **`B` must have full column rank.** Block methods orthonormalize the
  right-hand side, so linearly dependent columns (e.g. duplicated RHS) lead to a
  breakdown — give each column a distinct RHS.
- `memory` (in `KrylovWorkspaceOptions`) applies to `block_gmres` only; the
  solve-time tolerances (`atol`, `rtol`, `itmax`, `verbose`) come from
  `KrylovOptions` as usual.

```c
// Solve A X = B with p = 3 right-hand sides using block GMRES
int n = 100, p = 3;
double *B = /* n*p, column-major */, *X = malloc(sizeof(double) * n * p);

void *ws = NULL;
krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                              KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

KrylovOptions opts = krylov_default_options();
opts.atol = 1e-10; opts.rtol = 1e-10;
krylov_block_solve(ws, my_block_matvec, NULL, B, userdata, &opts);
krylov_block_get_X(ws, X, n, p);
krylov_block_workspace_free(ws);
```

See [Block Krylov methods](block_krylov.md) for the underlying algorithms.

## Building and linking

The library and headers are produced from the Krylov.jl repository with
[JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl), which embeds the Julia
runtime into a relocatable bundle.
The full, copy-pasteable build commands, plus the `gcc` / `gfortran` invocations and the platform-specific `-rpath` flags for Linux, macOS and Windows, live in [`interfaces/README.md`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/README.md).
Pre-built bundles are attached to each [release](https://github.com/JuliaSmoothOptimizers/Krylov.jl/releases).
