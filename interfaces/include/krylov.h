#ifndef KRYLOV_H
#define KRYLOV_H

/* Version */
#define KRYLOV_VERSION_MAJOR 0
#define KRYLOV_VERSION_MINOR 10
#define KRYLOV_VERSION_PATCH 7

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * libkrylov — C interface to Krylov.jl
 *
 * Typical use:
 *
 *   void *ws;
 *   krylov_workspace_create(KRYLOV_CG, n, n, KRYLOV_FLOAT64, KRYLOV_CPU, NULL, &ws);
 *
 *   KrylovOptions opts = krylov_default_options();
 *   opts.atol = 1e-10; opts.rtol = 1e-10;
 *   krylov_solve(ws, matvec_A, NULL, NULL, NULL, b, NULL, userdata, &opts);
 *
 *   krylov_get_x(ws, x, n);
 *   krylov_workspace_free(ws);
 *
 * Vectors (b, c, x, x0, and the callback buffers) are plain C arrays of the
 * element type selected by KrylovDataType: float, double, float _Complex or
 * double _Complex.  Full guide: https://jso.dev/Krylov.jl/dev/interfaces/overview/
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * Enumerators
 * ------------------------------------------------------------------------- */

typedef enum {
  KRYLOV_FLOAT32   = 0,
  KRYLOV_FLOAT64   = 1,
  KRYLOV_COMPLEX32 = 2,
  KRYLOV_COMPLEX64 = 3,
} KrylovDataType;

typedef enum {
  KRYLOV_CPU = 0,
} KrylovDeviceType;

typedef enum {
  KRYLOV_CG = 0,
  KRYLOV_CR = 1,
  KRYLOV_SYMMLQ = 2,
  KRYLOV_MINRES = 3,
  KRYLOV_MINRES_QLP = 4,
  KRYLOV_DIOM = 5,
  KRYLOV_DQGMRES = 6,
  KRYLOV_FOM = 7,
  KRYLOV_GMRES = 8,
  KRYLOV_FGMRES = 9,
  KRYLOV_BICGSTAB = 10,
  KRYLOV_CGS = 11,
  KRYLOV_BILQ = 12,
  KRYLOV_QMR = 13,
  KRYLOV_USYMLQ = 14,
  KRYLOV_USYMQR = 15,
  KRYLOV_TRICG = 16,
  KRYLOV_TRIMR = 17,
  KRYLOV_TRILQR = 18,
  KRYLOV_BILQR = 19,
  KRYLOV_LSLQ = 20,
  KRYLOV_LSQR = 21,
  KRYLOV_LSMR = 22,
  KRYLOV_USYMLQR = 23,
  KRYLOV_CGLS = 24,
  KRYLOV_CRLS = 25,
  KRYLOV_CGNE = 26,
  KRYLOV_CRMR = 27,
  KRYLOV_CRAIG = 28,
  KRYLOV_CRAIGMR = 29,
  KRYLOV_LNLQ = 30,
  KRYLOV_GPMR = 31,
  KRYLOV_CAR = 32,
  KRYLOV_MINARES = 33,
} KrylovSolverType;

typedef enum {
  KRYLOV_BLOCK_GMRES = 0,
  KRYLOV_BLOCK_MINRES = 1,
} KrylovBlockSolverType;

/* -------------------------------------------------------------------------
 * Callback types
 *
 * KrylovMatvec: computes y = A*x, y = A^H x, or applies the preconditioner
 *   y = M^-1 x (i.e. solves M y = x).
 *   x        : input vector  (read-only, length n)
 *   y        : output vector (write, length m)
 *   userdata : opaque pointer forwarded from krylov_solve
 *
 * KrylovBlockMatvec: block variant for block_gmres / block_minres.
 *   X        : input  block (read-only, n*p, column-major)
 *   Y        : output block (write,      m*p, column-major)
 *   p        : block size (number of columns)
 *   userdata : opaque pointer forwarded from krylov_block_solve
 * ------------------------------------------------------------------------- */

typedef void (*KrylovMatvec)(const void *x, void *y, void *userdata);
typedef void (*KrylovBlockMatvec)(const void *X, void *Y, int p, void *userdata);

/* -------------------------------------------------------------------------
 * API functions
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * Workspace options (construction-time)
 *
 * Passed to krylov_workspace_create.  These control how the workspace is
 * allocated, so they belong to creation rather than to the solve call.
 * Initialise with krylov_default_workspace_options() before overriding.
 * Sentinel 0 means "use solver default".
 *
 * Fields ignored by a given solver are silently disregarded.
 * ------------------------------------------------------------------------- */

typedef struct {
  int memory;  /* 0 → 20  (GMRES / FGMRES / FOM / DIOM / DQGMRES / GPMR)      */
  int window;  /* 0 → 5   (MINRES / SYMMLQ / LSQR / LSMR / LSLQ)              */
} KrylovWorkspaceOptions;

/* -------------------------------------------------------------------------
 * Solver options (solve-time)
 *
 * Passed to krylov_solve.  Initialise with krylov_default_options() before
 * overriding individual fields.  Sentinel values mean "use solver default":
 *   NaN  for double fields  (atol, rtol, tau, nu, timemax)
 *   0    for int fields     (itmax, restart, reorthogonalization, linesearch)
 *   0.0  for lambda/radius  (off, which is the default)
 *
 * Fields ignored by a given solver are silently disregarded.
 * ------------------------------------------------------------------------- */

typedef struct {
  double atol;                /* NaN → sqrt(eps(T)) per precision                              */
  double rtol;                /* NaN → sqrt(eps(T)) per precision                              */
  int    itmax;               /* 0   → solver default                                          */
  int    verbose;             /* 0   = silent                                                  */
  double lambda;              /* 0.0 = no regularisation/shift (LSQR/LSMR/CGLS/MINRES/...)     */
  double tau;                 /* NaN → solver default (TriCG / TriMR : 1.0)                    */
  double nu;                  /* NaN → solver default (TriCG / TriMR : -1.0)                   */
  double timemax;             /* NaN → Inf (no time limit), seconds; every solver             */
  double radius;              /* 0.0 = no trust region (CG/CR/CGLS/CRLS/LSQR/LSMR)             */
  int    restart;             /* 0/1 restart GMRES(k)/FGMRES/FOM/block_gmres (uses memory)     */
  int    reorthogonalization; /* 0/1 reorthogonalize basis (GMRES family, GPMR, block_gmres)   */
  int    linesearch;          /* 0/1 detect negative curvature (CG/CR/MINRES/MINRES-QLP)       */
} KrylovOptions;

/*
 * Create a workspace for `solver` on an m-by-n operator in precision `dtype`.
 * `device` is KRYLOV_CPU.  `wopts` may be NULL to use the defaults; the opaque
 * handle is written to *ws_out.
 * Returns 0 on success, -1 on error, -2 on an unknown (solver, dtype) pair.
 */
int krylov_workspace_create(KrylovSolverType solver, int m, int n, KrylovDataType dtype, KrylovDeviceType device, const KrylovWorkspaceOptions* wopts, void** ws_out);

/* Return a KrylovWorkspaceOptions with every field at its 0 "use default" sentinel. */
KrylovWorkspaceOptions krylov_default_workspace_options(void);

/* Return a KrylovOptions with every field at its NaN/0 "use default" sentinel. */
KrylovOptions krylov_default_options(void);

/*
 * Write the Krylov.jl version of this library into *major, *minor, *patch
 * (the same values as the KRYLOV_VERSION_* macros).
 */
void krylov_get_version(int* major, int* minor, int* patch);

/*
 * Solve the linear system with the workspace's solver.
 *   matvec_A  : computes y = A*x        (required)
 *   matvec_At : computes y = A^H x      (NULL unless the solver uses the adjoint,
 *                                        e.g. BiLQ, QMR, LSQR, LSMR, CGLS, CRAIG)
 *   matvec_M  : preconditioner (centered for symmetric solvers, left otherwise);
 *               must compute y = M^-1 x, i.e. solve M y = x  (NULL = none)
 *   matvec_N  : right preconditioner; must compute y = N^-1 x (solve N y = x).
 *               Used only by solvers that accept both sides (GMRES, FGMRES, FOM,
 *               DIOM, DQGMRES, BiCGSTAB, CGS, BiLQ, QMR); ignored otherwise.
 *               (NULL = no right preconditioner)
 *   b         : right-hand side, length m
 *   c         : second right-hand side, length n (NULL if not needed)
 *   userdata  : forwarded unchanged to every callback
 *   opts      : solve-time options, or NULL for the defaults
 * Returns 0 on success, -1 on error.
 */
int krylov_solve(void* ws, KrylovMatvec matvec_A, KrylovMatvec matvec_At, KrylovMatvec matvec_M, KrylovMatvec matvec_N, const void* b, const void* c, void* userdata, const KrylovOptions* opts);

/* Copy the primal solution (length n) into `x`.  Returns 0, or -1 on error. */
int krylov_get_x(void* ws, void* x, int n);

/*
 * Copy the second (dual) solution into `y`, for the two-solution solvers
 * (TriCG, TriMR, USYMLQR, GPMR, BiLQR, TriLQR, CRAIG, CRAIGMR, LNLQ).
 * Its length is the dual-solution size: n for TriCG/TriMR/USYMLQR, m otherwise.
 * Returns 0, -1 on error, or -2 if the solver has a single solution.
 */
int krylov_get_y(void* ws, void* y, int m);

/* Return 1 if the last solve converged, 0 if it did not, or -1 on error. */
int krylov_is_solved(void* ws);

/* Return the number of iterations performed, or -1 on error. */
int krylov_niter(void* ws);

/* Return the solve time in seconds, or -1.0 on error. */
double krylov_elapsed_time(void* ws);

/*
 * Set the initial guess (length n) for the next krylov_solve.
 * Returns 0, -1 on error, or -2 if the solver does not support warm starting.
 */
int krylov_warm_start(void* ws, const void* x0, int n);

/*
 * Set both initial guesses for the next krylov_solve, for the two-solution
 * solvers (TriCG, TriMR, GPMR, BiLQR, TriLQR, USYMLQR).
 *   x0 : primal guess, length nx (same size as krylov_get_x)
 *   y0 : dual guess,   length ny (same size as krylov_get_y)
 * Returns 0, -1 on error, or -2 if the solver has a single solution.
 */
int krylov_warm_start2(void* ws, const void* x0, const void* y0, int nx, int ny);

/*
 * Release the workspace; the handle must not be used afterwards.
 * Returns 0 on success, or 1 if the handle was not found.
 */
int krylov_workspace_free(void* ws);


/* -------------------------------------------------------------------------
 * Block Krylov interface (block_gmres / block_minres)
 *
 * The right-hand side B and the solution X are m-by-p / n-by-p blocks,
 * stored column-major.  Otherwise the workflow matches the single-vector
 * interface above (create -> solve -> get_X -> free).
 * ------------------------------------------------------------------------- */

/*
 * Create a block workspace for `solver` (KRYLOV_BLOCK_GMRES / KRYLOV_BLOCK_MINRES)
 * with block size p (number of right-hand sides).  `wopts` may be NULL.
 * Returns 0 on success, -1 on error, -2 on an unknown (solver, dtype) pair.
 */
int krylov_block_workspace_create(KrylovBlockSolverType solver, int m, int n, int p, KrylovDataType dtype, KrylovDeviceType device, const KrylovWorkspaceOptions* wopts, void** ws_out);

/*
 * Solve A X = B for the m-by-p block B.
 *   matvec_A : computes Y = A*X for a block of p columns (required)
 *   matvec_M : preconditioner; must compute Y = M^-1 X (solve M Y = X).
 *              Left for block_gmres, centered for block_minres  (NULL = none)
 *   matvec_N : right preconditioner; must compute Y = N^-1 X (solve N Y = X).
 *              Used only by block_gmres; ignored by block_minres.
 *              (NULL = no right preconditioner)
 *   B        : right-hand side block, m*p, column-major
 *   opts     : solve-time options, or NULL for the defaults
 * Returns 0 on success, -1 on error.
 */
int krylov_block_solve(void* ws, KrylovBlockMatvec matvec_A, KrylovBlockMatvec matvec_M, KrylovBlockMatvec matvec_N, const void* B, void* userdata, const KrylovOptions* opts);

/* Copy the n-by-p solution block (column-major) into `X`.  Returns 0, or -1 on error. */
int krylov_block_get_X(void* ws, void* X, int n, int p);

/* Return 1 if the last block solve converged, 0 if it did not, or -1 on error. */
int krylov_block_is_solved(void* ws);

/* Return the number of iterations performed, or -1 on error. */
int krylov_block_niter(void* ws);

/* Return the block solve time in seconds, or -1.0 on error. */
double krylov_block_elapsed_time(void* ws);

/*
 * Set the initial guess (n-by-p block) for the next block solve.
 * Returns 0, -1 on error, or -2 if the solver does not support warm starting.
 */
int krylov_block_warm_start(void* ws, const void* x0, int n, int p);

/* Release the block workspace.  Returns 0, or 1 if the handle was not found. */
int krylov_block_workspace_free(void* ws);


#ifdef __cplusplus
}
#endif

#endif /* KRYLOV_H */
