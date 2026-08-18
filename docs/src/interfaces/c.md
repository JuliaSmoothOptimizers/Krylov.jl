# [C interface](@id c-interface)

Include `krylov.h` and link against `libkrylov` (see [Building libkrylov](@ref building-libkrylov)).
The concepts (callbacks, data types, options, return codes, block solvers) are described in the [reference](@ref reference-interfaces).
This page collects runnable C programs.

## Example

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
    if (i > 0)      y[i] += A->off[i-1] * x[i-1];
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
  krylov_solve(ws, matvec_A, NULL, NULL, NULL, b, NULL, &A, &opts);
  krylov_get_x(ws, x, N);

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  for (int i = 0; i < N; i++) printf(" %.2f", x[i]);
  printf("\n");

  krylov_workspace_free(ws);
  return 0;
}
```

## Block example

Solve `A X = B` with several right-hand sides at once using block GMRES. The block `B` is `n×p`, column-major, and must have full column rank (see [Block Krylov solvers](@ref block-krylov-methods)). Switch `KRYLOV_BLOCK_GMRES` to `KRYLOV_BLOCK_MINRES` for block MINRES
([`examples/C/block_gmres.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/C/block_gmres.c)):

```c
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include "krylov.h"

#define N 16    /* operator dimension */
#define P 3     /* number of right-hand sides (block width) */

/* Block matvec:  Y = A * X  for a block of p columns (column-major). */
static void block_A(const void *Xv, void *Yv, int p, void *userdata)
{
  const double *X = (const double *)Xv;
  double       *Y = (double *)Yv;
  int n = *(const int *)userdata;
  for (int j = 0; j < p; j++) {
    const double *x = X + (size_t)j * n;
    double       *y = Y + (size_t)j * n;
    for (int i = 0; i < n; i++) {
      y[i] = 8.0 * x[i];
      if (i > 0)     y[i] -= x[i-1];
      if (i < n - 1) y[i] -= x[i+1];
    }
  }
}

int main(void)
{
  int n = N, p = P;

  /* X_true with independent columns, then B = A * X_true (column-major). */
  double Xtrue[N*P], B[N*P], X[N*P];
  for (int j = 0; j < p; j++)
    for (int i = 0; i < n; i++) {
      double t = (double)(i + 1) / n;
      Xtrue[i + (size_t)j*n] = (j == 0) ? 1.0 : (j == 1 ? t : t*t);
    }
  block_A(Xtrue, B, p, &n);

  void *ws = NULL;
  krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10;
  opts.rtol = 1e-10;
  krylov_block_solve(ws, block_A, NULL, NULL, B, &n, &opts);
  krylov_block_get_X(ws, X, n, p);

  double err = 0.0;
  for (int k = 0; k < n*p; k++) {
    double d = fabs(X[k] - Xtrue[k]);
    if (d > err) err = d;
  }
  printf("Block solved: %s   niter: %d   max error: %.1e\n",
         krylov_block_is_solved(ws) ? "yes" : "no",
         krylov_block_niter(ws), err);

  krylov_block_workspace_free(ws);
  return 0;
}
```

## Preconditioning

A preconditioner is just another matrix-free callback. But it applies the **action of the inverse**: it must compute `y = M⁻¹ x` (solve `M y = x`), never `y = M x`. The symmetric solvers (CG, CR, CAR, MINRES, …) take a single **centered** preconditioner `M`, which preserves the symmetry of the system. Only the non-symmetric solvers split it into a left `M` and a right `N`. Here is preconditioned CAR (a recent solver for symmetric systems) with a Jacobi preconditioner `M = diag(A)`, so `M⁻¹ x` is just `xᵢ / dᵢ`
([`examples/C/preconditioning.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/C/preconditioning.c)):

```c
#include <stdio.h>
#include "krylov.h"

#define N 8

typedef struct { int n; double diag[N]; double off; } Tri;

static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  const Tri    *A = (const Tri *)userdata;
  for (int i = 0; i < A->n; i++) {
    y[i] = A->diag[i] * x[i];
    if (i > 0)        y[i] += A->off * x[i-1];
    if (i < A->n - 1) y[i] += A->off * x[i+1];
  }
}

/* y = M⁻¹ x with M = diag(A): apply the INVERSE, i.e. divide by the diagonal. */
static void precond_M(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  const Tri    *A = (const Tri *)userdata;
  for (int i = 0; i < A->n; i++) y[i] = x[i] / A->diag[i];
}

int main(void)
{
  Tri A = { .n = N, .off = -1.0 };
  for (int i = 0; i < N; i++) A.diag[i] = 2.0 + (double)i;

  double x_true[N], b[N], x[N];
  for (int i = 0; i < N; i++) x_true[i] = 1.0;
  matvec_A(x_true, b, &A);

  void *ws = NULL;
  krylov_workspace_create(KRYLOV_CAR, N, N, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10; opts.rtol = 1e-10;

  // matvec_M = centered preconditioner (symmetric solvers use a single M); matvec_N is NULL.
  // GMRES, BiCGSTAB, QMR, ... instead use matvec_M as a left and matvec_N as a right preconditioner.
  krylov_solve(ws, matvec_A, NULL, precond_M, NULL, b, NULL, &A, &opts);
  krylov_get_x(ws, x, N);

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  krylov_workspace_free(ws);
  return 0;
}
```

## Least-squares (rectangular operator)

Least-squares and least-norm solvers (LSMR, LSQR, CGLS, CRAIG, …) reach the operator through **both** `A` and its adjoint `Aᴴ`, so they need a second callback. With a rectangular `A` of size `m × n` the buffer lengths differ: `matvec_A` maps length `n` to length `m`, while `matvec_At` maps length `m` to length `n`
([`examples/C/least_squares.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/examples/C/least_squares.c)):

```c
#include <stdio.h>
#include "krylov.h"

#define M 5   /* rows */
#define N 3   /* columns */

typedef struct { int m, n; const double *A; } Mat;  /* A row-major, m x n */

/* y = A * x   (x length n, y length m) */
static void matvec_A(const void *xv, void *yv, void *ud)
{
  const double *x = (const double *)xv; double *y = (double *)yv;
  const Mat *A = (const Mat *)ud;
  for (int i = 0; i < A->m; i++) {
    double s = 0.0;
    for (int j = 0; j < A->n; j++) s += A->A[i*A->n + j] * x[j];
    y[i] = s;
  }
}

/* y = Aᴴ x = Aᵀ x   (x length m, y length n) */
static void matvec_At(const void *xv, void *yv, void *ud)
{
  const double *x = (const double *)xv; double *y = (double *)yv;
  const Mat *A = (const Mat *)ud;
  for (int j = 0; j < A->n; j++) {
    double s = 0.0;
    for (int i = 0; i < A->m; i++) s += A->A[i*A->n + j] * x[i];
    y[j] = s;
  }
}

int main(void)
{
  static const double A[M*N] = { 1,0,0,  0,1,0,  0,0,1,  1,1,0,  0,1,1 };
  Mat mat = { M, N, A };

  double x_true[N] = {1.0, 2.0, 3.0}, b[M], x[N];
  matvec_A(x_true, b, &mat);                 // consistent RHS

  void *ws = NULL;
  krylov_workspace_create(KRYLOV_LSMR, M, N, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-12; opts.rtol = 1e-12;

  // LSMR needs the adjoint: pass matvec_At in the second slot.
  krylov_solve(ws, matvec_A, matvec_At, NULL, NULL, b, NULL, &mat, &opts);
  krylov_get_x(ws, x, N);                     // solution has length n

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  krylov_workspace_free(ws);
  return 0;
}
```

## More examples

The repository keeps several complete C programs, all compiled and run in CI:

- [`test/C/test_all_solvers.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_all_solvers.c), every solver, on SPD, non-symmetric and least-squares problems.
- [`test/C/test_api.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_api.c), options, preconditioner, warm start, workspace reuse, error codes, `Float32`.
- [`test/C/test_block.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_block.c), `block_gmres` and `block_minres`.
