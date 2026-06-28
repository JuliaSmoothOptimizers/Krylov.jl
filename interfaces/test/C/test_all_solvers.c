/*
 * test_all_solvers.c — tests all solvers accessible via the libkrylov C interface.
 *
 * Two test problems (n x n dense matrices stored as flat arrays):
 *   SPD:    A = tridiag(-1, 2, -1),  b = A * ones,  x_true = ones
 *   NONSYM: A = tridiag(-1, n, -1),  b = A * ones,  x_true = ones
 *   LS:     A = tridiag(-1, n, -1) with m > n,  b = A * ones
 *           (same matrix, rectangular variant for least-squares solvers)
 *
 * Compile (after building libkrylov with juliac — see interfaces/README.md):
 *   gcc -O2 -o test_all_solvers interfaces/test/C/test_all_solvers.c \
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
 * Problem sizes
 * ------------------------------------------------------------------------- */
#define N    20    /* square dimension */
#define M    30    /* number of rows for rectangular problems */

/* -------------------------------------------------------------------------
 * Dense matrix-vector product  y = A * x  (row-major, m×n)
 * ------------------------------------------------------------------------- */
static void matvec_dense(const double *A, int m, int n,
                          const double *x, double *y)
{
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++)
            y[i] += A[i * n + j] * x[j];
    }
}

/* -------------------------------------------------------------------------
 * Userdata struct passed to every callback
 * ------------------------------------------------------------------------- */
typedef struct {
    const double *A;   /* m×n matrix (row-major) */
    const double *At;  /* n×m matrix (row-major) */
    int m, n;
} MatData;

static void cb_A(const void *xv, void *yv, void *ud)
{
    const MatData *d = (const MatData *)ud;
    matvec_dense(d->A, d->m, d->n, (const double *)xv, (double *)yv);
}

static void cb_At(const void *xv, void *yv, void *ud)
{
    const MatData *d = (const MatData *)ud;
    matvec_dense(d->At, d->n, d->m, (const double *)xv, (double *)yv);
}

/* -------------------------------------------------------------------------
 * Build an n×n tridiagonal matrix into a flat row-major array.
 * diag_val on the main diagonal, off_val on the sub/super diagonals.
 * ------------------------------------------------------------------------- */
static void make_tridiag(double *A, int n, double diag_val, double off_val)
{
    memset(A, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++) {
        A[i * n + i] = diag_val;
        if (i > 0)   A[i * n + (i-1)] = off_val;
        if (i < n-1) A[i * n + (i+1)] = off_val;
    }
}

/* Build a rectangular m×n version of the same pattern (first n rows). */
static void make_tridiag_rect(double *A, int m, int n, double diag_val, double off_val)
{
    memset(A, 0, sizeof(double) * m * n);
    for (int i = 0; i < m; i++) {
        if (i < n) A[i * n + i] = diag_val;
        if (i > 0 && i-1 < n)   A[i * n + (i-1)] = off_val;
        if (i < n-1)             A[i * n + (i+1)] = off_val;
    }
}

/* Transpose an m×n matrix into n×m. */
static void transpose(const double *A, double *At, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];
}

/* Infinity norm of a vector. */
static double norm_inf(const double *v, int n)
{
    double r = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (a > r) r = a;
    }
    return r;
}

/* -------------------------------------------------------------------------
 * Solver metadata
 * ------------------------------------------------------------------------- */
typedef enum {
    PROB_SPD,    /* symmetric positive definite (square system, x_true = ones) */
    PROB_SPD_QD, /* quasi-definite saddle-point (tricg/trimr), x=y=ones */
    PROB_NONSYM, /* square non-symmetric */
    PROB_GPMR,   /* gpmr: [I A; B I][x;y]=[b;c], x=y=ones, B passed via matvec_At slot */
    PROB_LS,     /* rectangular least-squares */
} ProblemKind;

typedef struct {
    KrylovSolverType solver;
    const char      *name;
    ProblemKind      prob;
    int              need_At;  /* 1 = pass cb_At, 0 = pass NULL */
    int              has_y;    /* 1 = krylov_get_y returns a valid dual */
    int              need_c;   /* 1 = pass second RHS c (size n), 0 = NULL */
} SolverInfo;

static const SolverInfo SOLVERS[] = {
    /* symmetric / Hermitian */
    { KRYLOV_CG,         "cg",         PROB_SPD,    0, 0, 0 },
    { KRYLOV_CR,         "cr",         PROB_SPD,    0, 0, 0 },
    { KRYLOV_SYMMLQ,     "symmlq",     PROB_SPD,    0, 0, 0 },
    { KRYLOV_MINRES,     "minres",     PROB_SPD,    0, 0, 0 },
    { KRYLOV_MINRES_QLP, "minres_qlp", PROB_SPD,    0, 0, 0 },
    { KRYLOV_CAR,        "car",        PROB_SPD,    0, 0, 0 },
    { KRYLOV_MINARES,    "minares",    PROB_SPD,    0, 0, 0 },
    { KRYLOV_TRICG,      "tricg",      PROB_SPD_QD, 1, 1, 1 },
    { KRYLOV_TRIMR,      "trimr",      PROB_SPD_QD, 1, 1, 1 },
    /* non-symmetric square */
    { KRYLOV_BICGSTAB,   "bicgstab",   PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_CGS,        "cgs",        PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_BILQ,       "bilq",       PROB_NONSYM, 1, 0, 0 },
    { KRYLOV_QMR,        "qmr",        PROB_NONSYM, 1, 0, 0 },
    { KRYLOV_DIOM,       "diom",       PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_DQGMRES,    "dqgmres",    PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_FOM,        "fom",        PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_GMRES,      "gmres",      PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_FGMRES,     "fgmres",     PROB_NONSYM, 0, 0, 0 },
    { KRYLOV_BILQR,      "bilqr",      PROB_NONSYM, 1, 1, 1 },
    { KRYLOV_TRILQR,     "trilqr",     PROB_NONSYM, 1, 1, 1 },
    { KRYLOV_USYMLQ,     "usymlq",     PROB_NONSYM, 1, 0, 1 },
    { KRYLOV_USYMQR,     "usymqr",     PROB_NONSYM, 1, 0, 1 },
    { KRYLOV_USYMLQR,    "usymlqr",    PROB_NONSYM, 1, 1, 1 },
    /* gpmr: fptr_At carries B, need_c=1, has_y=1 */
    { KRYLOV_GPMR,       "gpmr",       PROB_GPMR,   1, 1, 1 },
    /* least-squares / least-norm */
    { KRYLOV_LSLQ,       "lslq",       PROB_LS,     1, 0, 0 },
    { KRYLOV_LSQR,       "lsqr",       PROB_LS,     1, 0, 0 },
    { KRYLOV_LSMR,       "lsmr",       PROB_LS,     1, 0, 0 },
    { KRYLOV_CGLS,       "cgls",       PROB_LS,     1, 0, 0 },
    { KRYLOV_CRLS,       "crls",       PROB_LS,     1, 0, 0 },
    { KRYLOV_CGNE,       "cgne",       PROB_LS,     1, 0, 0 },
    { KRYLOV_CRMR,       "crmr",       PROB_LS,     1, 0, 0 },
    { KRYLOV_CRAIG,      "craig",      PROB_LS,     1, 0, 0 },
    { KRYLOV_CRAIGMR,    "craigmr",    PROB_LS,     1, 0, 0 },
    { KRYLOV_LNLQ,       "lnlq",       PROB_LS,     1, 0, 0 },
};

#define N_SOLVERS ((int)(sizeof(SOLVERS) / sizeof(SOLVERS[0])))

/* -------------------------------------------------------------------------
 * Run one solver test.  Returns 1 on pass, 0 on failure.
 * ------------------------------------------------------------------------- */
static int run_test(const SolverInfo *info,
                    MatData *spd_data,  const double *spd_b,    const double *spd_c,
                    const double *spd_qd_b, const double *spd_qd_c,
                    MatData *nsym_data, const double *nsym_b,   const double *nsym_c,
                    const double *gpmr_b,   const double *gpmr_c,
                    MatData *ls_data,   const double *ls_b)
{
    MatData      *data;
    const double *b, *c;
    int m, n;
    double tol = 1e-6;

    switch (info->prob) {
    case PROB_SPD:    data = spd_data;  b = spd_b;     c = spd_c;     m = N; n = N; break;
    case PROB_SPD_QD: data = spd_data;  b = spd_qd_b;  c = spd_qd_c;  m = N; n = N; break;
    case PROB_NONSYM: data = nsym_data; b = nsym_b;    c = nsym_c;    m = N; n = N; break;
    case PROB_GPMR:   data = nsym_data; b = gpmr_b;    c = gpmr_c;    m = N; n = N; break;
    default:          data = ls_data;   b = ls_b;      c = NULL;      m = M; n = N; break;
    }

    void *ws = NULL;
    KrylovWorkspaceOptions wopts = krylov_default_workspace_options();
    int ret = krylov_workspace_create(info->solver, m, n,
                                      KRYLOV_FLOAT64, KRYLOV_CPU, &wopts, &ws);
    if (ret != 0) {
        printf("  FAIL  krylov_workspace_create returned %d\n", ret);
        return 0;
    }

    KrylovOptions opts = krylov_default_options();
    opts.atol = 1e-8;
    opts.rtol = 1e-8;

    ret = krylov_solve(ws,
                       cb_A,
                       info->need_At ? cb_At : NULL,
                       NULL,
                       b,
                       info->need_c ? c : NULL,
                       data,
                       &opts);
    if (ret != 0) {
        printf("  FAIL  krylov_solve returned %d\n", ret);
        krylov_workspace_free(ws);
        return 0;
    }

    if (!krylov_is_solved(ws)) {
        printf("  FAIL  did not converge (niter=%d)\n", krylov_niter(ws));
        krylov_workspace_free(ws);
        return 0;
    }

    /* Check primal solution x ≈ ones */
    double x[N];
    krylov_get_x(ws, x, n);
    double err = 0.0;
    for (int i = 0; i < n; i++) err += (x[i] - 1.0) * (x[i] - 1.0);
    err = sqrt(err) / sqrt((double)n);
    if (err > tol) {
        printf("  FAIL  ||x - x_true|| / sqrt(n) = %.3e  (tol=%.3e)\n", err, tol);
        krylov_workspace_free(ws);
        return 0;
    }

    /* For solvers with a dual solution, just check finiteness */
    if (info->has_y) {
        double y[M];
        int ret_y = krylov_get_y(ws, y, m);
        if (ret_y != 0) {
            printf("  FAIL  krylov_get_y returned %d\n", ret_y);
            krylov_workspace_free(ws);
            return 0;
        }
        if (!isfinite(norm_inf(y, m))) {
            printf("  FAIL  y contains non-finite values\n");
            krylov_workspace_free(ws);
            return 0;
        }
    }

    krylov_workspace_free(ws);
    return 1;
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    double ones_N[N];
    for (int i = 0; i < N; i++) ones_N[i] = 1.0;

    /* ---- SPD problem: A = tridiag(-1,2,-1), b = A*ones, x_true = ones ---- */
    /* c = A^T * ones = b  (A is symmetric) */
    double spd_A[N*N], spd_At[N*N], spd_b[N], spd_c[N];
    make_tridiag(spd_A, N, 2.0, -1.0);
    transpose(spd_A, spd_At, N, N);
    matvec_dense(spd_A, N, N, ones_N, spd_b);
    matvec_dense(spd_At, N, N, ones_N, spd_c);  /* = spd_b, A is symmetric */
    MatData spd_data = { spd_A, spd_At, N, N };

    /* ---- Quasi-definite saddle-point for tricg/trimr (tau=1, nu=-1) ------- */
    /* System: [I  A][x] = [b_qd]   with x_true = y_true = ones             */
    /*         [A' -I][y] = [c_qd]                                           */
    /* => b_qd = x + A*y = ones + spd_b,  c_qd = A'*x - y = spd_b - ones   */
    double spd_qd_b[N], spd_qd_c[N];
    for (int i = 0; i < N; i++) {
        spd_qd_b[i] = ones_N[i] + spd_b[i];   /* ones + A*ones */
        spd_qd_c[i] = spd_b[i]  - ones_N[i];  /* A'*ones - ones = A*ones - ones */
    }

    /* ---- Non-symmetric: A = tridiag(-1,N,-1), b = A*ones, c = A^T*ones --- */
    double nsym_A[N*N], nsym_At[N*N], nsym_b[N], nsym_c[N];
    make_tridiag(nsym_A, N, (double)N, -1.0);
    transpose(nsym_A, nsym_At, N, N);
    matvec_dense(nsym_A, N, N, ones_N, nsym_b);
    matvec_dense(nsym_At, N, N, ones_N, nsym_c);
    MatData nsym_data = { nsym_A, nsym_At, N, N };

    /* ---- GPMR: [I A; B I][x;y]=[b;c], B=A^T, x_true=y_true=ones --------- */
    /* b = ones + A*ones,  c = A^T*ones + ones */
    double gpmr_b[N], gpmr_c[N];
    for (int i = 0; i < N; i++) {
        gpmr_b[i] = ones_N[i] + nsym_b[i];
        gpmr_c[i] = nsym_c[i] + ones_N[i];
    }

    /* ---- Least-squares: A is M×N tridiag, b = A*ones, x_true = ones ------ */
    double ls_A[M*N], ls_At[N*M], ls_b[M];
    make_tridiag_rect(ls_A, M, N, (double)N, -1.0);
    transpose(ls_A, ls_At, M, N);
    matvec_dense(ls_A, M, N, ones_N, ls_b);
    MatData ls_data = { ls_A, ls_At, M, N };

    /* ---- Run all solver tests -------------------------------------------- */
    int n_pass = 0, n_fail = 0;
    for (int i = 0; i < N_SOLVERS; i++) {
        const SolverInfo *info = &SOLVERS[i];
        printf("%-14s ... ", info->name);
        fflush(stdout);
        if (run_test(info, &spd_data, spd_b, spd_c, spd_qd_b, spd_qd_c, &nsym_data, nsym_b, nsym_c, gpmr_b, gpmr_c, &ls_data, ls_b)) {
            printf("PASS  (niter omitted)\n");
            n_pass++;
        } else {
            n_fail++;
        }
    }

    printf("\n%d/%d passed\n", n_pass, N_SOLVERS);
    return n_fail > 0 ? 1 : 0;
}
