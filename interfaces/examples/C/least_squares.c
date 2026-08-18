/*
 * least_squares.c — solve an overdetermined least-squares problem with LSMR.
 *
 * Minimizes ‖b - A x‖₂ for a rectangular A of size m x n (m > n).  LSMR (like
 * LSQR, CGLS, CRAIG, ...) reaches the operator through BOTH A and its adjoint,
 * so it needs TWO callbacks:
 *
 *   matvec_A  : y = A * x     (x has length n, y has length m)
 *   matvec_At : y = Aᴴ * x    (x has length m, y has length n)   <-- the adjoint
 *
 * This is the main difference from the square solvers (CG/GMRES), where m == n
 * and the adjoint is usually unused (pass NULL).  Here the lengths differ, so be
 * careful which dimension each buffer has.
 *
 * The matrix is stored row-major (A[i*n + j]); A is consistent (b = A * x_true)
 * so LSMR recovers x_true exactly.
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *
 *   gcc -O2 -o least_squares interfaces/examples/C/least_squares.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libkrylov.so \
 *       -Wl,-rpath,'$ORIGIN/../lib/julia'
 *
 * Expected output:
 *   Solved: yes   niter: ...
 *   x = [ 1.00  2.00  3.00 ]
 */

#include <stdio.h>

#include "krylov.h"

#define M 5   /* rows    (number of equations)  */
#define N 3   /* columns (number of unknowns)   */

typedef struct { int m, n; const double *A; } Mat;  /* A is row-major, m x n */

/* y = A * x      (x length n, y length m) */
static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  const Mat    *M_ = (const Mat *)userdata;
  for (int i = 0; i < M_->m; i++) {
    double s = 0.0;
    for (int j = 0; j < M_->n; j++) s += M_->A[i * M_->n + j] * x[j];
    y[i] = s;
  }
}

/* y = Aᴴ * x = Aᵀ * x   (real matrix; x length m, y length n) */
static void matvec_At(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  const Mat    *M_ = (const Mat *)userdata;
  for (int j = 0; j < M_->n; j++) {
    double s = 0.0;
    for (int i = 0; i < M_->m; i++) s += M_->A[i * M_->n + j] * x[i];
    y[j] = s;
  }
}

int main(void)
{
  /* A (5 x 3), full column rank, stored row-major. */
  static const double A[M * N] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 1.0,
  };
  Mat mat = { M, N, A };

  double x_true[N] = {1.0, 2.0, 3.0};
  double b[M], x[N];
  matvec_A(x_true, b, &mat);          /* consistent RHS: b = A * x_true */

  /* Operator is m x n: pass (m, n) = (M, N) at creation. */
  void *ws = NULL;
  if (krylov_workspace_create(KRYLOV_LSMR, M, N, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws) != 0)
    return 1;

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-12;
  opts.rtol = 1e-12;

  /* LSMR needs the adjoint: pass matvec_At in the second slot. */
  int ret = krylov_solve(ws, matvec_A, matvec_At, NULL, NULL, b, NULL, &mat, &opts);
  if (ret != 0) { krylov_workspace_free(ws); return 1; }

  krylov_get_x(ws, x, N);             /* solution has length n */

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  printf("x = [");
  for (int j = 0; j < N; j++) printf(" %.2f", x[j]);
  printf(" ]\n");

  krylov_workspace_free(ws);
  return 0;
}
