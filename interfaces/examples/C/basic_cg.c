/*
 * basic_cg.c — minimal example: solve a 5x5 SPD system with CG.
 *
 * A = tridiag(-1, 2, -1),  b = [1, 0, 0, 0, 1]^T
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *
 *   gcc -o basic_cg interfaces/examples/C/basic_cg.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libkrylov.so \
 *       -Wl,-rpath,'$ORIGIN/../lib/julia'
 *
 * Expected output:
 *   Solved: yes   niter: 3   time: ...
 *   x = [ 1.00  1.00  1.00  1.00  1.00 ]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "krylov.h"

/* -------------------------------------------------------------------------
 * Problem data
 * ------------------------------------------------------------------------- */

#define N 5

/* Tridiagonal matrix stored as three diagonals for simplicity. */
typedef struct {
  int    n;
  double diag[N];   /* main diagonal */
  double off[N-1];  /* sub/super diagonal */
} TriDiag;

/* Matvec callback:  y = A * x  */
static void matvec_A(const void *xv, void *yv, void *userdata)
{
  const double *x  = (const double *)xv;
  double       *y  = (double *)yv;
  const TriDiag *A = (const TriDiag *)userdata;
  int n = A->n;

  for (int i = 0; i < n; i++) {
    y[i] = A->diag[i] * x[i];
    if (i > 0)   y[i] += A->off[i-1] * x[i-1];
    if (i < n-1) y[i] += A->off[i]   * x[i+1];
  }
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(void)
{
  /* Build A = tridiag(-1, 2, -1) */
  TriDiag A;
  A.n = N;
  for (int i = 0; i < N;   i++) A.diag[i] = 2.0;
  for (int i = 0; i < N-1; i++) A.off[i]  = -1.0;

  /* Right-hand side */
  double b[N] = {1.0, 0.0, 0.0, 0.0, 1.0};

  /* Solution buffer */
  double x[N];

  /* -----------------------------------------------------------------------
   * Create workspace for CG, double precision, CPU
   * --------------------------------------------------------------------- */
  void *ws = NULL;
  int ret = krylov_workspace_create(KRYLOV_CG, N, N,
                                    KRYLOV_FLOAT64, KRYLOV_CPU,
                                    NULL,    /* workspace options (NULL = defaults) */
                                    &ws);
  if (ret != 0) {
    fprintf(stderr, "krylov_workspace_create failed (%d)\n", ret);
    return 1;
  }

  /* -----------------------------------------------------------------------
   * Solve
   * --------------------------------------------------------------------- */
  KrylovOptions opts = krylov_default_options();
  opts.atol = 1e-10;
  opts.rtol = 1e-10;

  ret = krylov_solve(ws,
                     matvec_A,   /* y = A*x */
                     NULL,       /* y = A'*x  (CG doesn't need it) */
                     NULL,       /* no left preconditioner */
                     NULL,       /* no right preconditioner */
                     b,          /* right-hand side b (size m) */
                     NULL,       /* c = NULL  (CG only needs one RHS) */
                     &A,         /* userdata forwarded to matvec_A */
                     &opts);     /* solver options (NULL = all defaults) */
  if (ret != 0) {
    fprintf(stderr, "krylov_solve failed (%d)\n", ret);
    krylov_workspace_free(ws);
    return 1;
  }

  /* -----------------------------------------------------------------------
   * Retrieve results
   * --------------------------------------------------------------------- */
  ret = krylov_get_x(ws, x, N);
  if (ret != 0) {
    fprintf(stderr, "krylov_get_x failed (%d)\n", ret);
    krylov_workspace_free(ws);
    return 1;
  }

  printf("Solved: %s   niter: %d   time: %.3e s\n",
         krylov_is_solved(ws) ? "yes" : "no",
         krylov_niter(ws),
         krylov_elapsed_time(ws));

  printf("x = [");
  for (int i = 0; i < N; i++)
    printf(" %.2f", x[i]);
  printf(" ]\n");

  /* -----------------------------------------------------------------------
   * Free workspace
   * --------------------------------------------------------------------- */
  krylov_workspace_free(ws);

  return 0;
}
