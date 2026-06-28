# [C interface](@id c-interface)

Include `krylov.h` and link against `libkrylov` (see [Building libkrylov](@ref building-libkrylov)).
The concepts (callbacks, data types, options, return codes, block solvers) are described in the interfaces [overview](@ref overview-interfaces).
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

## Block example

Solve `A X = B` with several right-hand sides using block GMRES. The block `B` is `n×p`, column-major, and must have full column rank (see [Block Krylov solvers](@ref block-krylov-methods):

```c
// Y = A * X for a block of p columns (column-major)
void block_A(const void *Xv, void *Yv, int p, void *userdata) { /* fill Y = A*X */ }

int n = 100, p = 3;
double *B = /* n*p, column-major */, *X = malloc(sizeof(double) * n * p);

void *ws = NULL;
krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                              KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

KrylovOptions opts = krylov_default_options();
opts.atol = 1e-10; opts.rtol = 1e-10;
krylov_block_solve(ws, block_A, NULL, B, userdata, &opts);
krylov_block_get_X(ws, X, n, p);
krylov_block_workspace_free(ws);
```

## More examples

The repository keeps several complete C programs, all compiled and run in CI:

- [`test/C/test_all_solvers.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_all_solvers.c), every solver, on SPD, non-symmetric and least-squares problems.
- [`test/C/test_api.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_api.c), options, preconditioner, warm start, workspace reuse, error codes, `Float32`.
- [`test/C/test_block.c`](https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/interfaces/test/C/test_block.c), `block_gmres` and `block_minres`.
