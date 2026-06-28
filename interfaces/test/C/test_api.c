/*
 * test_api.c — behavioural tests for the libkrylov C interface.
 *
 * Complements test_all_solvers.c (which checks convergence of every solver)
 * by exercising the parts of the API that file does not touch:
 *   - default option structs and their sentinels
 *   - workspace options actually taking effect (memory / window)
 *   - preconditioning (matvec_M)
 *   - warm starting
 *   - workspace reuse across several solves
 *   - error codes (unknown solver, get_y on a 1-solution solver, double free)
 *   - a non-double precision (Float32)
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *   gcc -O2 -o test_api interfaces/test/C/test_api.c \
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

/* -------------------------------------------------------------------------
 * Tiny test harness
 * ------------------------------------------------------------------------- */
static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg)                                       \
    do {                                                       \
        if (cond) {                                            \
            n_pass++;                                          \
        } else {                                              \
            n_fail++;                                          \
            printf("  FAIL  %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                      \
    } while (0)

/* -------------------------------------------------------------------------
 * Problem: A = tridiag(-1, 2, -1), SPD, x_true = ones, b = A * ones.
 * Stored as three diagonals; the same struct is the callback userdata.
 * ------------------------------------------------------------------------- */
typedef struct { int n; double diag; double off; } Tri;

/* y = A * x  (double) */
static void matvec_A(const void *xv, void *yv, void *ud)
{
    const double *x = (const double *)xv;
    double       *y = (double *)yv;
    const Tri    *A = (const Tri *)ud;
    for (int i = 0; i < A->n; i++) {
        y[i] = A->diag * x[i];
        if (i > 0)        y[i] += A->off * x[i-1];
        if (i < A->n - 1) y[i] += A->off * x[i+1];
    }
}

/* y = A * x  (float / Float32) */
static void matvec_A_f32(const void *xv, void *yv, void *ud)
{
    const float *x = (const float *)xv;
    float       *y = (float *)yv;
    const Tri   *A = (const Tri *)ud;
    for (int i = 0; i < A->n; i++) {
        y[i] = (float)A->diag * x[i];
        if (i > 0)        y[i] += (float)A->off * x[i-1];
        if (i < A->n - 1) y[i] += (float)A->off * x[i+1];
    }
}

/* y = M^{-1} * x  with M = diag(A) = 2*I  (Jacobi preconditioner) */
static void precond_M(const void *xv, void *yv, void *ud)
{
    const double *x = (const double *)xv;
    double       *y = (double *)yv;
    const Tri    *A = (const Tri *)ud;
    for (int i = 0; i < A->n; i++) y[i] = x[i] / A->diag;
}

static double rel_err_ones(const double *x, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; i++) e += (x[i] - 1.0) * (x[i] - 1.0);
    return sqrt(e) / sqrt((double)n);
}

/* Fill b = A * ones (double). */
static void rhs_from_ones(const Tri *A, double *b)
{
    double *ones = (double *)malloc(sizeof(double) * A->n);
    for (int i = 0; i < A->n; i++) ones[i] = 1.0;
    matvec_A(ones, b, (void *)A);
    free(ones);
}

/* =========================================================================
 * Tests
 * ========================================================================= */

static void test_default_options(void)
{
    printf("default options ...\n");
    KrylovOptions o = krylov_default_options();
    CHECK(isnan(o.atol),  "default atol is NaN sentinel");
    CHECK(isnan(o.rtol),  "default rtol is NaN sentinel");
    CHECK(o.itmax == 0,   "default itmax is 0");
    CHECK(o.verbose == 0, "default verbose is 0");
    CHECK(o.lambda == 0.0,"default lambda is 0");

    KrylovWorkspaceOptions w = krylov_default_workspace_options();
    CHECK(w.memory == 0, "default memory is 0 sentinel");
    CHECK(w.window == 0, "default window is 0 sentinel");
}

static void test_unknown_solver(void)
{
    printf("unknown solver ...\n");
    void *ws = NULL;
    int ret = krylov_workspace_create((KrylovSolverType)999, 4, 4,
                                      KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    CHECK(ret == -2, "unknown (solver,dtype) returns -2");
    CHECK(ws == NULL, "handle left untouched on failure");
}

static void test_null_options(void)
{
    printf("NULL options ...\n");
    const int n = 16;
    Tri A = { n, 2.0, -1.0 };
    double b[16], x[16];
    rhs_from_ones(&A, b);

    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    int ret = krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, NULL);
    CHECK(ret == 0,            "solve with NULL opts succeeds");
    CHECK(krylov_is_solved(ws),"solve with NULL opts converges");
    krylov_get_x(ws, x, n);
    CHECK(rel_err_ones(x, n) < 1e-6, "NULL opts solution is correct");
    krylov_workspace_free(ws);
}

static void test_get_y_single_solution(void)
{
    printf("get_y on single-solution solver ...\n");
    const int n = 8;
    Tri A = { n, 2.0, -1.0 };
    double b[8], y[8];
    rhs_from_ones(&A, b);

    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, NULL);
    int ret = krylov_get_y(ws, y, n);
    CHECK(ret == -2, "get_y returns -2 when there is no dual solution");
    krylov_workspace_free(ws);
}

static void test_double_free(void)
{
    printf("double free ...\n");
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, 4, 4, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    CHECK(krylov_workspace_free(ws) == 0, "first free returns 0");
    CHECK(krylov_workspace_free(ws) == 1, "second free returns 1 (not found)");
}

static void test_preconditioner(void)
{
    printf("preconditioner (Jacobi) ...\n");
    const int n = 32;
    Tri A = { n, 2.0, -1.0 };
    double b[32], x[32];
    rhs_from_ones(&A, b);

    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10;
    int ret = krylov_solve(ws, matvec_A, NULL, precond_M, b, NULL, &A, &o);
    CHECK(ret == 0,             "preconditioned solve succeeds");
    CHECK(krylov_is_solved(ws), "preconditioned solve converges");
    krylov_get_x(ws, x, n);
    CHECK(rel_err_ones(x, n) < 1e-6, "preconditioned solution is correct");
    krylov_workspace_free(ws);
}

static void test_reuse_workspace(void)
{
    printf("workspace reuse (two RHS) ...\n");
    const int n = 16;
    Tri A = { n, 2.0, -1.0 };
    double b1[16], b2[16], x[16];
    rhs_from_ones(&A, b1);            /* x_true = ones */
    for (int i = 0; i < n; i++) b2[i] = 2.0 * b1[i]; /* x_true = 2*ones */

    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);

    krylov_solve(ws, matvec_A, NULL, NULL, b1, NULL, &A, NULL);
    krylov_get_x(ws, x, n);
    CHECK(krylov_is_solved(ws) && rel_err_ones(x, n) < 1e-6, "first solve correct");

    krylov_solve(ws, matvec_A, NULL, NULL, b2, NULL, &A, NULL);
    krylov_get_x(ws, x, n);
    double e = 0.0;
    for (int i = 0; i < n; i++) e += (x[i] - 2.0) * (x[i] - 2.0);
    CHECK(krylov_is_solved(ws) && sqrt(e)/sqrt((double)n) < 1e-6, "second solve (reused ws) correct");
    krylov_workspace_free(ws);
}

static void test_warm_start(void)
{
    printf("warm start ...\n");
    const int n = 32;
    Tri A = { n, 2.0, -1.0 };
    double b[32], x_star[32];
    rhs_from_ones(&A, b);

    /* Cold solve from zero. */
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, NULL);
    int niter_cold = krylov_niter(ws);
    krylov_get_x(ws, x_star, n);
    krylov_workspace_free(ws);

    /* Warm solve seeded with the (nearly exact) solution. */
    krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
    int wret = krylov_warm_start(ws, x_star, n);
    CHECK(wret == 0, "warm_start accepted by CG");
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, NULL);
    int niter_warm = krylov_niter(ws);
    CHECK(krylov_is_solved(ws),        "warm-started solve converges");
    CHECK(niter_warm < niter_cold,     "warm start reduces iteration count");
    krylov_workspace_free(ws);
}

static void test_workspace_memory(void)
{
    printf("workspace option: memory ...\n");
    const int n = 24;
    Tri A = { n, 2.0, -1.0 };
    double b[24], x[24];
    rhs_from_ones(&A, b);

    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10; o.itmax = 2000;

    /* DQGMRES is genuinely limited-memory: `memory` is the truncation parameter
       (only the `memory` most recent basis vectors are orthogonalized against).
       A small value must still drive the solve to the correct solution. */
    KrylovWorkspaceOptions wsmall = krylov_default_workspace_options();
    wsmall.memory = 4;
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_DQGMRES, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &wsmall, &ws);
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, &o);
    krylov_get_x(ws, x, n);
    CHECK(krylov_is_solved(ws) && rel_err_ones(x, n) < 1e-6,
          "DQGMRES with memory=4 converges");
    krylov_workspace_free(ws);

    /* For GMRES, `memory` sizes the initial storage (the basis grows on demand);
       a small hint must not affect correctness. */
    KrylovWorkspaceOptions whint = krylov_default_workspace_options();
    whint.memory = 5;
    krylov_workspace_create(KRYLOV_GMRES, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &whint, &ws);
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, &o);
    krylov_get_x(ws, x, n);
    CHECK(krylov_is_solved(ws) && rel_err_ones(x, n) < 1e-6,
          "GMRES with memory=5 hint converges");
    krylov_workspace_free(ws);
}

static void test_workspace_window(void)
{
    printf("workspace option: window (MINRES) ...\n");
    const int n = 24;
    Tri A = { n, 2.0, -1.0 };
    double b[24], x[24];
    rhs_from_ones(&A, b);

    KrylovOptions o = krylov_default_options();
    o.atol = 1e-10; o.rtol = 1e-10;

    /* window = 1 must not break the residual-estimation machinery. */
    KrylovWorkspaceOptions w = krylov_default_workspace_options();
    w.window = 1;
    void *ws = NULL;
    krylov_workspace_create(KRYLOV_MINRES, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, &w, &ws);
    krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, &A, &o);
    krylov_get_x(ws, x, n);
    CHECK(krylov_is_solved(ws) && rel_err_ones(x, n) < 1e-6, "MINRES with window=1 converges");
    krylov_workspace_free(ws);
}

static void test_float32(void)
{
    printf("Float32 precision ...\n");
    const int n = 16;
    Tri A = { n, 2.0, -1.0 };
    float ones[16], b[16], x[16];
    for (int i = 0; i < n; i++) ones[i] = 1.0f;
    matvec_A_f32(ones, b, &A);

    void *ws = NULL;
    int ret = krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT32, KRYLOV_CPU, NULL, &ws);
    CHECK(ret == 0, "Float32 workspace created");
    ret = krylov_solve(ws, matvec_A_f32, NULL, NULL, b, NULL, &A, NULL);
    CHECK(ret == 0 && krylov_is_solved(ws), "Float32 CG converges");
    krylov_get_x(ws, x, n);
    float e = 0.0f;
    for (int i = 0; i < n; i++) e += (x[i] - 1.0f) * (x[i] - 1.0f);
    CHECK(sqrtf(e) / sqrtf((float)n) < 1e-3f, "Float32 solution is correct");
    krylov_workspace_free(ws);
}

int main(void)
{
    test_default_options();
    test_unknown_solver();
    test_null_options();
    test_get_y_single_solution();
    test_double_free();
    test_preconditioner();
    test_reuse_workspace();
    test_warm_start();
    test_workspace_memory();
    test_workspace_window();
    test_float32();

    printf("\n%d checks passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
