/*
 * test_block.c — tests for the block Krylov C interface (block_gmres / block_minres).
 *
 * Problem: A = tridiag(-1, 8, -1) (SPD, strongly diagonally dominant), n x n.
 * The right-hand side is an n x p block B = A * X_true, where X_true has
 * linearly independent columns (block Krylov methods require a full-rank block).
 * Blocks are column-major: entry (i, j) is at index i + j*n.
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *   gcc -O2 -o test_block interfaces/test/C/test_block.c \
 *       -I interfaces/build/include interfaces/build/lib/libkrylov.so \
 *       -Wl,-rpath,'$ORIGIN/../lib/julia' -lm
 *
 * Exit code: 0 if all tests pass, 1 otherwise.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "krylov.h"

#define N 20
#define P 3

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { n_pass++; }                                     \
        else { n_fail++;                                            \
            printf("  FAIL  %s  (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

/* y = A * X for a block of p columns, A = tridiag(-1, 8, -1), column-major. */
static void block_A(const void *Xv, void *Yv, int p, void *ud)
{
    const double *X = (const double *)Xv;
    double       *Y = (double *)Yv;
    int n = *(const int *)ud;
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

/* Y = M^{-1} X with M = diag(A) = 8*I  (Jacobi preconditioner). */
static void block_M(const void *Xv, void *Yv, int p, void *ud)
{
    const double *X = (const double *)Xv;
    double       *Y = (double *)Yv;
    int n = *(const int *)ud;
    for (int k = 0; k < n * p; k++) Y[k] = X[k] / 8.0;
}

/* X_true with independent columns: col 0 = 1, col 1 = i/n, col 2 = (i/n)^2. */
static void fill_xtrue(double *X, int n, int p)
{
    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / n;
            X[i + (size_t)j * n] = (j == 0) ? 1.0 : (j == 1 ? t : t * t);
        }
}

static double block_err(const double *X, const double *Xtrue, int n, int p)
{
    double e = 0.0;
    for (int k = 0; k < n * p; k++) {
        double d = X[k] - Xtrue[k];
        if (fabs(d) > e) e = fabs(d);
    }
    return e;
}

/* Build B = A * Xtrue using the same operator as the callback. */
static void make_rhs(const double *Xtrue, double *B, int n, int p)
{
    block_A(Xtrue, B, p, &n);
}

static void test_block_solver(KrylovBlockSolverType solver, const char *name)
{
    printf("%s ...\n", name);
    int n = N, p = P;
    double Xtrue[N*P], B[N*P], X[N*P];
    fill_xtrue(Xtrue, n, p);
    make_rhs(Xtrue, B, n, p);

    void *ws = NULL;
    int ret = krylov_block_workspace_create(solver, n, n, p,
                                            KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    CHECK(ret == 0, "block workspace created");

    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10; o.itmax = 200;
    ret = krylov_block_solve(ws, block_A, NULL, NULL, B, &n, &o);
    CHECK(ret == 0,                   "block solve returns 0");
    CHECK(krylov_block_is_solved(ws), "block solve converged");
    CHECK(krylov_block_niter(ws) > 0, "block niter is positive");
    CHECK(krylov_block_elapsed_time(ws) >= 0.0, "block elapsed_time is non-negative");

    ret = krylov_block_get_X(ws, X, n, p);
    CHECK(ret == 0, "block_get_X returns 0");
    CHECK(block_err(X, Xtrue, n, p) < 1e-6, "block solution is correct");

    krylov_block_workspace_free(ws);
}

static void test_block_preconditioner(void)
{
    printf("block_gmres + Jacobi preconditioner ...\n");
    int n = N, p = P;
    double Xtrue[N*P], B[N*P], X[N*P];
    fill_xtrue(Xtrue, n, p);
    make_rhs(Xtrue, B, n, p);

    void *ws = NULL;
    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10; o.itmax = 200;
    int ret = krylov_block_solve(ws, block_A, block_M, NULL, B, &n, &o);
    CHECK(ret == 0 && krylov_block_is_solved(ws), "preconditioned block solve converges");
    krylov_block_get_X(ws, X, n, p);
    CHECK(block_err(X, Xtrue, n, p) < 1e-6, "preconditioned block solution is correct");
    krylov_block_workspace_free(ws);

    /* Right preconditioner: block_gmres accepts matvec_N. */
    printf("block_gmres + right Jacobi preconditioner ...\n");
    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    ret = krylov_block_solve(ws, block_A, NULL, block_M, B, &n, &o);
    CHECK(ret == 0 && krylov_block_is_solved(ws), "right-preconditioned block solve converges");
    krylov_block_get_X(ws, X, n, p);
    CHECK(block_err(X, Xtrue, n, p) < 1e-6, "right-preconditioned block solution is correct");
    krylov_block_workspace_free(ws);
}

static void test_block_warm_start(void)
{
    printf("block_gmres warm start ...\n");
    int n = N, p = P;
    double Xtrue[N*P], B[N*P];
    fill_xtrue(Xtrue, n, p);
    make_rhs(Xtrue, B, n, p);

    void *ws = NULL;
    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10; o.itmax = 200;

    krylov_block_solve(ws, block_A, NULL, NULL, B, &n, &o);
    int niter_cold = krylov_block_niter(ws);
    krylov_block_workspace_free(ws);

    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    int wret = krylov_block_warm_start(ws, Xtrue, n, p);
    CHECK(wret == 0, "block warm_start accepted");
    krylov_block_solve(ws, block_A, NULL, NULL, B, &n, &o);
    CHECK(krylov_block_is_solved(ws), "warm-started block solve converges");
    CHECK(krylov_block_niter(ws) < niter_cold, "block warm start reduces iterations");
    krylov_block_workspace_free(ws);
}

static void test_block_memory(void)
{
    printf("block_gmres memory option ...\n");
    int n = N, p = P;
    double Xtrue[N*P], B[N*P], X[N*P];
    fill_xtrue(Xtrue, n, p);
    make_rhs(Xtrue, B, n, p);

    KrylovWorkspaceOptions w = krylov_default_workspace_options();
    w.memory = 4;
    void *ws = NULL;
    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, &w, &ws);
    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10; o.itmax = 200;
    krylov_block_solve(ws, block_A, NULL, NULL, B, &n, &o);
    krylov_block_get_X(ws, X, n, p);
    CHECK(krylov_block_is_solved(ws) && block_err(X, Xtrue, n, p) < 1e-6,
          "block_gmres with memory=4 converges");
    krylov_block_workspace_free(ws);
}

static void test_block_errors(void)
{
    printf("block error codes ...\n");
    int n = N, p = P;
    void *ws = NULL;
    int ret = krylov_block_workspace_create((KrylovBlockSolverType)99, n, n, p,
                                            KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    CHECK(ret == -2, "unknown block solver returns -2");

    krylov_block_workspace_create(KRYLOV_BLOCK_GMRES, n, n, p,
                                  KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    CHECK(krylov_block_workspace_free(ws) == 0, "first block free returns 0");
    CHECK(krylov_block_workspace_free(ws) == 1, "second block free returns 1");
}

int main(void)
{
    test_block_solver(KRYLOV_BLOCK_GMRES,  "block_gmres");
    test_block_solver(KRYLOV_BLOCK_MINRES, "block_minres");
    test_block_preconditioner();
    test_block_warm_start();
    test_block_memory();
    test_block_errors();

    printf("\n%d checks passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
