/*
 * block_gmres.c — solve A X = B with several right-hand sides at once.
 *
 * A = tridiag(-1, 8, -1) (SPD),  B = A * X_true  with independent columns.
 * Blocks are n x p, column-major.  Switch KRYLOV_BLOCK_GMRES to
 * KRYLOV_BLOCK_MINRES below to use block MINRES on the same system.
 *
 * Compile (after building libkrylov — see interfaces/README.md):
 *   gcc -o block_gmres interfaces/examples/C/block_gmres.c \
 *       -I interfaces/build/include interfaces/build/lib/libkrylov.so \
 *       -Wl,-rpath,'$ORIGIN/../lib/julia' -lm
 *
 * Expected output:
 *   Block solved: yes   niter: ...   max error: ...
 */

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

  /* Create a block workspace, solve, retrieve the n x p solution block. */
  void *ws = NULL;
  krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10;
  opts.rtol = 1e-10;
  krylov_block_solve(ws, block_A, NULL, B, &n, &opts);
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
