# [Reference](@id reference-interfaces)

Detailed reference for the `libkrylov` C and Fortran API. See the [overview](@ref overview-interfaces) for the big picture and a quickstart, and the [C interface](@ref c-interface) / [Fortran interface](@ref fortran-interface) pages for runnable programs.

## [API overview](@id ref-api-overview)

```c
// Callback signature: computes y = A * x, y = Aᴴ * x, or y = M \ x
typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);

// Library version (same values as the KRYLOV_VERSION_* macros)
void krylov_get_version(int *major, int *minor, int *patch);

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
                 KrylovMatvec matvec_M,   // left/centered preconditioner, NULL = none
                 KrylovMatvec matvec_N,   // right preconditioner,         NULL = none
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
int    krylov_warm_start2(void *ws, const void *x0, const void *y0, int nx, int ny);
int    krylov_workspace_free(void *ws);
```

## [Workspace lifetime](@id ref-lifetime)

`krylov_workspace_create` is the only allocating call, and `krylov_workspace_free` the only releasing one. Between them you may call `krylov_solve` and the accessors as many times as you like. For example, solve `A x = b₁` then `A x = b₂` with the same operator, reusing all the internal vectors. The handle becomes invalid the moment you free it; do not touch it afterwards. Free each workspace exactly once to avoid leaking the (potentially large) internal storage.

## [The matvec callbacks](@id ref-callbacks)

A callback receives three raw pointers and returns nothing:

```c
void matvec(const void *x, void *y, void *userdata);
```

- `x` is the input vector (read-only). `y` is the output vector you must fill. `userdata` is the opaque pointer you handed to `krylov_solve`, forwarded unchanged to every callback; use it to carry whatever your product needs (the matrix, problem sizes, scratch buffers).
- Both vectors point to arrays of the element type you chose at creation (`double`, `float`, `double _Complex`, `float _Complex`). Cast them accordingly before use.
- The library owns `x` and `y`. Do not keep, free, or resize them. They are valid only for the duration of the call.

The four slots play different roles, and their vector lengths follow the operator shape (`m` rows, `n` columns):

| Slot        | Computes      | `x` length | `y` length | When to pass |
|-------------|---------------|:----------:|:----------:|--------------|
| `matvec_A`  | `y = A * x`   | `n`        | `m`        | always (required) |
| `matvec_At` | `y = Aᴴ * x`  | `m`        | `n`        | solvers that use the adjoint |
| `matvec_M`  | `y = M⁻¹ * x` | `m`        | `m`        | preconditioning (centered for symmetric solvers, left otherwise) |
| `matvec_N`  | `y = N⁻¹ * x` | `n`        | `n`        | right preconditioning (non-symmetric solvers) |

For square systems `m == n`, so all these lengths coincide. This is the common case (CG, GMRES). The distinction matters for least-squares and least-norm solvers, where `A` is rectangular.

!!! note "GPMR is special"
    For `KRYLOV_GPMR` the second callback slot is not the adjoint of `A`. It is a separate operator `B` (size `n × m`) used by the block formulation `[0 A; B 0]`. Pass the product of `B` where `matvec_At` would normally go.

## [Data types and precision](@id ref-datatypes)

A workspace is created for exactly one element type, selected with `KrylovDataType`:

| Enum               | Julia type   | C element type    |
|--------------------|--------------|-------------------|
| `KRYLOV_FLOAT32`   | `Float32`    | `float`           |
| `KRYLOV_FLOAT64`   | `Float64`    | `double`          |
| `KRYLOV_COMPLEX32` | `ComplexF32` | `float _Complex`  |
| `KRYLOV_COMPLEX64` | `ComplexF64` | `double _Complex` |

Every buffer that crosses the boundary (`b`, `c`, the callback's `x` and `y`, `x0`, and the destinations of `krylov_get_x` / `krylov_get_y`) must be an array of that element type. The complex layout is the standard interleaved real/imaginary one. It is identical to C99 `_Complex` and to Julia's `Complex{T}`, so no conversion is needed. The only device currently available is `KRYLOV_CPU = 0`.

## [Choosing a solver](@id ref-choosing)

`KrylovSolverType` lists every exposed solver (`KRYLOV_CG`, `KRYLOV_GMRES`, and so on). Three properties decide how you call `krylov_solve`.

**Does it need `matvec_At`?** Pass `NULL` for the left column, a real callback for the right one:

| Pass `NULL` for `matvec_At` | Provide `matvec_At` |
|-----------------------------|---------------------|
| CG, CR, CAR, MINRES, MINRES-QLP, MINARES, SYMMLQ, GMRES, FGMRES, FOM, DIOM, DQGMRES, BiCGSTAB, CGS | BiLQ, QMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR, TriCG, TriMR, LSLQ, LSQR, LSMR, CGLS, CRLS, CGNE, CRMR, CRAIG, CRAIGMR, LNLQ, GPMR\* |

\* GPMR uses that slot for `B`, not `Aᴴ` (see the note above).

**Does it take a second right-hand side `c`?** Required by TriCG, TriMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR and GPMR. Pass `NULL` for `c` otherwise.

**Does it produce a second solution `y`?** Retrieve it with `krylov_get_y` for the two-solution solvers TriCG, TriMR, USYMLQR, GPMR, BiLQR, TriLQR, CRAIG, CRAIGMR and LNLQ (its length is the size of the dual solution: `n` for TriCG/TriMR/USYMLQR, `m` for the others). For every other solver `krylov_get_y` returns `-2`.

## [Options](@id ref-options)

Options are split by *when* they are consumed:

| Struct | Passed to | Fields |
|--------|-----------|--------|
| `KrylovWorkspaceOptions` | `krylov_workspace_create` | `memory`, `window` |
| `KrylovOptions` | `krylov_solve` | `atol`, `rtol`, `itmax`, `verbose`, `lambda`, `tau`, `nu`, `timemax`, `radius`, `restart`, `reorthogonalization`, `linesearch` |

Always initialise a struct from its `krylov_default_*` helper, then override only the fields you need. Every field has a sentinel (`0` for ints, `NaN` for doubles) meaning *use the solver default*. Fields irrelevant to a given solver are silently ignored, so it is safe to set, say, `lambda` even when running CG.

**Construction-time** (`KrylovWorkspaceOptions`). These size internal storage, so they belong to creation rather than to the solve:

- `memory`, default `20`. Its meaning depends on the solver:
  - **DIOM** and **DQGMRES** are genuinely limited-memory: `memory` is the number of recent basis vectors a new vector is orthogonalized against (the truncation parameter), so it directly changes the algorithm.
  - **GMRES**, **FGMRES**, **FOM** and **GPMR** are full methods here: `memory` only sizes the initial storage and the basis grows on demand, so it is a performance hint, not an algorithmic knob (use `restart` for the restarted GMRES(k) variant).
- `window`, the number of past residuals used by the residual-norm estimate of MINRES, SYMMLQ, LSQR, LSMR and LSLQ, default `5`.

**Solve-time** (`KrylovOptions`):

- `atol`, `rtol`, the absolute and relative stopping tolerances. `NaN` falls back to `√eps(T)` for the chosen precision.
- `itmax`, the iteration cap. `0` falls back to the solver default (a small multiple of the problem size).
- `timemax`, a wall-clock budget in seconds; the solve stops once it is exceeded. `NaN` means no limit (`Inf`). Honoured by every solver.
- `verbose`. `0` is silent, a positive value prints convergence information every `verbose` iterations.
- `lambda`, a shift or regularisation; `0.0` means none. For the least-squares and least-norm solvers (LSQR, LSMR, CGLS, CRLS, LNLQ, LSLQ, CRAIG, CRAIGMR) it is the Tikhonov parameter. For the symmetric solvers (MINRES, MINRES-QLP, SYMMLQ, MINARES) it shifts the system to `(A + lambda·I) x = b`.
- `radius`, a trust-region constraint `‖x‖ ≤ radius` for CG, CR, CGLS, CRLS, LSQR and LSMR. `0.0` (default) means unconstrained; with `radius > 0` the step stops on the trust-region boundary. Useful inside optimization solvers.
- `linesearch`, set to `1` to detect nonpositive curvature and stop on it (CG, CR, MINRES, MINRES-QLP). Cannot be combined with `radius > 0`.
- `restart`, set to `1` to restart GMRES, FGMRES, FOM (and `block_gmres`) every `memory` iterations, i.e. the restarted GMRES(k) variant, with `k = memory` from `KrylovWorkspaceOptions`.
- `reorthogonalization`, set to `1` to reorthogonalize the Krylov basis (GMRES, FGMRES, FOM, DIOM, DQGMRES, GPMR, `block_gmres`), trading work for numerical robustness.
- `tau`, `nu`, the diagonal scalings of the saddle-point system solved by TriCG and TriMR. `NaN` falls back to `1.0` and `-1.0` respectively.

```c
// DQGMRES truncated to the 10 most recent Krylov basis vectors
KrylovWorkspaceOptions wopts = krylov_default_workspace_options();
wopts.memory = 10;
krylov_workspace_create(KRYLOV_DQGMRES, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &wopts, &ws);
```

## [Return codes](@id ref-return-codes)

Every function returns an `int` status (`krylov_elapsed_time` returns a `double`). The conventions are:

| Function | Success | Other values |
|----------|:-------:|--------------|
| `krylov_workspace_create` | `0` | `-1` internal error, `-2` unknown `(solver, precision)` combination |
| `krylov_solve` | `0` | `-1` internal error (for example a callback threw) |
| `krylov_get_x` | `0` | `-1` error (bad handle or size) |
| `krylov_get_y` | `0` | `-1` error, `-2` solver has a single solution |
| `krylov_warm_start` | `0` | `-1` error, `-2` solver does not support warm start |
| `krylov_warm_start2` | `0` | `-1` error, `-2` solver has a single solution |
| `krylov_workspace_free` | `0` | `1` handle not found |
| `krylov_is_solved` | `1` converged | `0` not converged, `-1` error |
| `krylov_niter` | iteration count | `-1` error |
| `krylov_elapsed_time` | seconds (`double`) | `-1.0` error |

Always check the status of `krylov_workspace_create` and `krylov_solve` before reading results, and use `krylov_is_solved` to confirm convergence. A non-zero `krylov_solve` means the run failed, whereas `krylov_is_solved == 0` means it ran but did not reach the tolerance within `itmax`.

## [Error handling and safety](@id ref-errors)

The return code is the reliable signal, so always check it. Any exception raised on the Julia side during a call is caught and converted to `-1`; it is never propagated across the C boundary. The offending call is logged to `stderr`. After a solve, `krylov_is_solved` separates a genuine failure (`krylov_solve` returned nonzero) from a run that did not converge within `itmax` (`krylov_solve` returned `0`, but `krylov_is_solved` is `0`).

Two classes of mistakes are the caller's responsibility and are **not** caught:

- **Wrong buffer sizes.** The library trusts the sizes you pass: `krylov_solve` reads `b` and the callback buffers assuming the dimensions given at creation. A buffer shorter than expected is an out-of-bounds access (undefined behavior), not a clean error.
- **A crash inside a callback.** The matvec and preconditioner callbacks are your own native code; a segfault or out-of-bounds write there takes down the whole process, and Julia cannot intercept it.

**Thread safety.** The library is not thread-safe; workspaces are tracked in shared global tables. So `krylov_workspace_create` and `krylov_workspace_free` must not run concurrently with each other or with any other `libkrylov` call. Solving on separate workspaces from separate threads is safe only if no workspace is being created or freed at the same time (and if your own callbacks are thread-safe).

## [Warm start](@id ref-warm-start)

To seed a solve with an initial guess `x0`, call `krylov_warm_start(ws, x0, n)` *before* `krylov_solve`. The guess is copied into the workspace and consumed by the next solve. Solvers that do not support warm starting return `-2`.

The two-solution solvers (TriCG, TriMR, GPMR, BiLQR, TriLQR, USYMLQR) take a guess for *both* unknowns through `krylov_warm_start2(ws, x0, y0, nx, ny)`. Here `x0` has the size of the primal solution (`krylov_get_x`) and `y0` the size of the dual one (`krylov_get_y`). Calling it on a single-solution solver returns `-2`. See [Warm-start](@ref warm-start) for the underlying mechanism.

## [Preconditioning](@id ref-preconditioning)

A preconditioner is passed through the `matvec_M` / `matvec_N` slots. A preconditioner callback applies the *action of the inverse*: it computes `y = M⁻¹ x` (that is, solves `M y = x`), never the forward product. Pass `NULL` in a slot for no preconditioner. Like the operator, preconditioners are matrix-free: they can be an incomplete factorization, a sparse approximate inverse, a multigrid V-cycle, or anything else you can evaluate. How the slots are interpreted depends on the solver:

- **Symmetric solvers** (CG, CR, CAR, MINRES, MINRES-QLP, SYMMLQ, MINARES) take a single **centered** preconditioner in `matvec_M`, a symmetric positive-definite `M` that preserves the symmetry of the system. `matvec_N` is ignored, so pass `NULL`. The callback sees vectors of length `n` (`= m`, the system is square).
- **Non-symmetric solvers** (GMRES, FGMRES, FOM, DIOM, DQGMRES, BiCGSTAB, CGS, BiLQ, QMR; and `block_gmres`) take a **left** preconditioner in `matvec_M` and a **right** one in `matvec_N` (both length `n`, the system is square). Supplying both gives a split preconditioner.

### Least-squares and least-norm solvers

These rectangular solvers follow Krylov.jl's convention of two preconditioners `M = E⁻¹` (acting on the `m`-dimensional data space) and `N = F⁻¹` (the `n`-dimensional solution space):

| Solvers | `matvec_M` | `matvec_N` |
|---------|-----------|-----------|
| LSQR, LSMR, LSLQ, CRAIG, CRAIGMR, LNLQ | `E⁻¹`, length `m` | `F⁻¹`, length `n` |
| CGLS, CRLS (normal equations) | single precond., length `n` | — |
| CGNE, CRMR (normal equations) | — | single precond., length `n` |

For LSQR/LSMR/LSLQ/CRAIG/CRAIGMR/LNLQ, `M` weights the residual norm and `N` the solution norm. The normal-equations methods (CGLS/CRLS, CGNE/CRMR) operate directly on `AᴴA` or `AAᴴ`. They take a single preconditioner on the `n`-space: through `matvec_M` for CGLS/CRLS, through `matvec_N` for CGNE/CRMR.

!!! note "GPMR and the two-RHS solvers"
    Preconditioning for GPMR (which has its own `C, D, E, F` operators) and for the two-RHS solvers (TriCG, TriMR, BiLQR, TriLQR, USYMLQ, USYMQR, USYMLQR) is not exposed through `matvec_M` / `matvec_N` yet; those slots are ignored for them.

See [Preconditioners](@ref preconditioners) for guidance on choosing one.

## [Block Krylov solvers](@id ref-block)

`block_gmres` and `block_minres` solve a system with several right-hand sides at once, `A X = B`, where `B` and `X` are `m×p` and `n×p` blocks. They have their own parallel API, mirroring the single-vector one:

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
                          KrylovBlockMatvec matvec_M,  // left preconditioner,  NULL = none
                          KrylovBlockMatvec matvec_N,  // right preconditioner, NULL = none
                          const void *B, void *userdata, const KrylovOptions *opts);
int    krylov_block_get_X(void *ws, void *X, int n, int p);
int    krylov_block_is_solved(void *ws);
int    krylov_block_niter(void *ws);
double krylov_block_elapsed_time(void *ws);
int    krylov_block_warm_start(void *ws, const void *X0, int n, int p);
int    krylov_block_workspace_free(void *ws);
```

Key differences from the single-vector API:

- **Blocks are column-major**, exactly like a Fortran or LAPACK matrix: entry `(i, j)` of an `n×p` block is at offset `i + j*n`. `B`, `X`, `X0`, and the callback's `X` and `Y` all use this layout.
- **The callback receives the block width `p`**, so it can apply `A` to all columns in one call (`Y = A·X`).
- **`B` must have full column rank.** Block methods orthonormalize the right-hand side, so linearly dependent columns (for example a duplicated RHS) lead to a breakdown. Give each column a distinct RHS.
- `memory` (in `KrylovWorkspaceOptions`) applies to `block_gmres` only. The solve-time tolerances (`atol`, `rtol`, `itmax`, `verbose`) come from `KrylovOptions` as usual.
- `matvec_M` (left preconditioner) is available for both block solvers; `matvec_N` (right) is used by `block_gmres` only and ignored by `block_minres`. Pass `NULL` for an unpreconditioned side.

Runnable block programs are shown on the [C interface](@ref c-interface) and [Fortran interface](@ref fortran-interface) pages.
See [Block Krylov methods](@ref block-krylov-methods) for the underlying algorithms.
