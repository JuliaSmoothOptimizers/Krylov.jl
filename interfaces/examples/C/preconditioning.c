/*
 * preconditioning.c — preconditioned CAR with a Jacobi preconditioner.
 *
 * CAR is a recent Krylov method for symmetric positive definite systems. Like
 * CG and CR, it uses a single CENTERED preconditioner M (a symmetric positive
 * definite operator that preserves the symmetry of the system).
 *
 * Solves A x = b with A = tridiag(-1, d_i, -1) where the diagonal d_i varies,
 * so the Jacobi preconditioner M = diag(A) is non-trivial.
 *
 * KEY POINT: the preconditioner callback applies the *action of the inverse*,
 * i.e. it must compute y = M^-1 x (solve M y = x), NOT y = M x.  For Jacobi,
 * M = diag(d_i) so M^-1 x is simply x_i / d_i.
 *
 * The preconditioner always goes in the matvec_M slot. For the symmetric solvers
 * it is the centered preconditioner (a single M, as here); the non-symmetric
 * solvers (GMRES, FGMRES, BiCGSTAB, QMR, ...) instead read matvec_M as a LEFT
 * preconditioner and additionally accept a RIGHT one in the matvec_N slot.
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *
 *   gcc -O2 -o preconditioning interfaces/examples/C/preconditioning.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libkrylov.so \
 *       -Wl,-rpath,'$ORIGIN/../lib/julia'
 *
 * Expected output:
 *   Solved: yes   niter: ...
 *   x = [ 1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00 ]
 */

#include <stdio.h>

#include "krylov.h"

#define N 8

/* Tridiagonal matrix A = tridiag(-1, d_i, -1) with a varying diagonal. */
typedef struct {
  int    n;
  double diag[N];   /* main diagonal d_i */
  double off;       /* sub/super diagonal (constant -1) */
} Tri;

/* y = A * x */
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

/* y = M^-1 * x  with the Jacobi preconditioner M = diag(A): y_i = x_i / d_i */
static void precond_M(const void *xv, void *yv, void *userdata)
{
  const double *x = (const double *)xv;
  double       *y = (double *)yv;
  const Tri    *A = (const Tri *)userdata;
  for (int i = 0; i < A->n; i++)
    y[i] = x[i] / A->diag[i];
}

int main(void)
{
  Tri A = { .n = N, .off = -1.0 };
  for (int i = 0; i < N; i++) A.diag[i] = 2.0 + (double)i;  /* varying diagonal */

  /* Right-hand side chosen so that the exact solution is all ones. */
  double x_true[N], b[N], x[N];
  for (int i = 0; i < N; i++) x_true[i] = 1.0;
  matvec_A(x_true, b, &A);

  void *ws = NULL;
  if (krylov_workspace_create(KRYLOV_CAR, N, N, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws) != 0)
    return 1;

  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10;
  opts.rtol = 1e-10;

  /* matvec_At = NULL (CAR does not use the adjoint),
     matvec_M  = precond_M (centered preconditioner, applies M^-1),
     matvec_N  = NULL (symmetric solver: a single preconditioner). */
  int ret = krylov_solve(ws, matvec_A, NULL, precond_M, NULL, b, NULL, &A, &opts);
  if (ret != 0) { krylov_workspace_free(ws); return 1; }

  krylov_get_x(ws, x, N);

  printf("Solved: %s   niter: %d\n",
         krylov_is_solved(ws) ? "yes" : "no", krylov_niter(ws));
  printf("x = [");
  for (int i = 0; i < N; i++) printf(" %.2f", x[i]);
  printf(" ]\n");

  krylov_workspace_free(ws);
  return 0;
}
