#!/usr/bin/env julia
# Generate interfaces/include/krylov.h from the function_sigs table in LibKrylov.jl
# and the SOLVERS table in generate_stores.jl (single source of truth).
# Usage:  julia interfaces/scripts/generate_header.jl

using Krylov

include(joinpath(@__DIR__, "solver_table.jl"))
include(joinpath(@__DIR__, "..", "src", "LibKrylov.jl"))

# Grab the signature table that LibKrylov.jl populated inside its module
function_sigs = LibKrylov.function_sigs

# ---------------------------------------------------------------------------
# Per-function documentation, emitted as a C comment before each prototype.
# Keep these in sync with the doc comments in include/krylov.f90.
# ---------------------------------------------------------------------------
const FUNCTION_DOCS = Dict{String,String}(
  "krylov_workspace_create" =>
    "Create a workspace for `solver` on an m-by-n operator in precision `dtype`.\n" *
    "`device` is KRYLOV_CPU.  `wopts` may be NULL to use the defaults; the opaque\n" *
    "handle is written to *ws_out.\n" *
    "Returns 0 on success, -1 on error, -2 on an unknown (solver, dtype) pair.",
  "krylov_default_workspace_options" =>
    "Return a KrylovWorkspaceOptions with every field at its 0 \"use default\" sentinel.",
  "krylov_default_options" =>
    "Return a KrylovOptions with every field at its NaN/0 \"use default\" sentinel.",
  "krylov_solve" =>
    "Solve the linear system with the workspace's solver.\n" *
    "  matvec_A  : computes y = A*x        (required)\n" *
    "  matvec_At : computes y = A^H x      (NULL unless the solver uses the adjoint,\n" *
    "                                       e.g. BiLQ, QMR, LSQR, LSMR, CGLS, CRAIG)\n" *
    "  matvec_M  : preconditioner; must compute y = M^-1 x, i.e. solve M y = x\n" *
    "              (NULL = no preconditioner)\n" *
    "  b         : right-hand side, length m\n" *
    "  c         : second right-hand side, length n (NULL if not needed)\n" *
    "  userdata  : forwarded unchanged to every callback\n" *
    "  opts      : solve-time options, or NULL for the defaults\n" *
    "Returns 0 on success, -1 on error.",
  "krylov_get_x" =>
    "Copy the primal solution (length n) into `x`.  Returns 0, or -1 on error.",
  "krylov_get_y" =>
    "Copy the dual solution (length m) into `y`, for two-solution solvers\n" *
    "(TriCG, TriMR, GPMR, BiLQR, TriLQR).\n" *
    "Returns 0, -1 on error, or -2 if the solver has a single solution.",
  "krylov_is_solved" =>
    "Return 1 if the last solve converged, 0 if it did not, or -1 on error.",
  "krylov_niter" =>
    "Return the number of iterations performed, or -1 on error.",
  "krylov_elapsed_time" =>
    "Return the solve time in seconds, or -1.0 on error.",
  "krylov_warm_start" =>
    "Set the initial guess (length n) for the next krylov_solve.\n" *
    "Returns 0, -1 on error, or -2 if the solver does not support warm starting.",
  "krylov_workspace_free" =>
    "Release the workspace; the handle must not be used afterwards.\n" *
    "Returns 0 on success, or 1 if the handle was not found.",
  "krylov_block_workspace_create" =>
    "Create a block workspace for `solver` (KRYLOV_BLOCK_GMRES / KRYLOV_BLOCK_MINRES)\n" *
    "with block size p (number of right-hand sides).  `wopts` may be NULL.\n" *
    "Returns 0 on success, -1 on error, -2 on an unknown (solver, dtype) pair.",
  "krylov_block_solve" =>
    "Solve A X = B for the m-by-p block B.\n" *
    "  matvec_A : computes Y = A*X for a block of p columns (required)\n" *
    "  matvec_M : preconditioner; must compute Y = M^-1 X (solve M Y = X)\n" *
    "             (NULL = no preconditioner)\n" *
    "  B        : right-hand side block, m*p, column-major\n" *
    "  opts     : solve-time options, or NULL for the defaults\n" *
    "Returns 0 on success, -1 on error.",
  "krylov_block_get_X" =>
    "Copy the n-by-p solution block (column-major) into `X`.  Returns 0, or -1 on error.",
  "krylov_block_is_solved" =>
    "Return 1 if the last block solve converged, 0 if it did not, or -1 on error.",
  "krylov_block_niter" =>
    "Return the number of iterations performed, or -1 on error.",
  "krylov_block_elapsed_time" =>
    "Return the block solve time in seconds, or -1.0 on error.",
  "krylov_block_warm_start" =>
    "Set the initial guess (n-by-p block) for the next block solve.\n" *
    "Returns 0, -1 on error, or -2 if the solver does not support warm starting.",
  "krylov_block_workspace_free" =>
    "Release the block workspace.  Returns 0, or 1 if the handle was not found.",
)

# Emit a C comment block (single line as /* ... */, multi-line as a /* * */ block).
function emit_doc(io, text)
  lines = split(text, '\n')
  if length(lines) == 1
    println(io, "/* $(lines[1]) */")
  else
    println(io, "/*")
    for l in lines
      println(io, isempty(l) ? " *" : " * $l")
    end
    println(io, " */")
  end
end

# ---------------------------------------------------------------------------
# Build KrylovSolverType enum entries from solver_table.jl (single source of truth)
# ---------------------------------------------------------------------------
solver_enum_entries = join(
  ["  $(enum_name) = $(si-1)" for (si, (_, _, enum_name)) in enumerate(SOLVERS)],
  ",\n"
)

block_solver_enum_entries = join(
  ["  $(enum_name) = $(si-1)" for (si, (_, _, enum_name)) in enumerate(BLOCK_SOLVERS)],
  ",\n"
)

# ---------------------------------------------------------------------------
# Write krylov.h
# ---------------------------------------------------------------------------
out_path = joinpath(@__DIR__, "..", "include", "krylov.h")
mkpath(dirname(out_path))

open(out_path, "w") do io
  println(io, """
#ifndef KRYLOV_H
#define KRYLOV_H

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
 *   krylov_solve(ws, matvec_A, NULL, NULL, b, NULL, userdata, &opts);
 *
 *   krylov_get_x(ws, x, n);
 *   krylov_workspace_free(ws);
 *
 * Vectors (b, c, x, x0, and the callback buffers) are plain C arrays of the
 * element type selected by KrylovDataType: float, double, float _Complex or
 * double _Complex.  Full guide: https://jso.dev/Krylov.jl/dev/c_fortran/
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
$(solver_enum_entries),
} KrylovSolverType;

typedef enum {
$(block_solver_enum_entries),
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
""")

  # Option structs must appear before any function that references them.
  println(io, """
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
 *   NaN  for double fields  (atol, rtol, tau, nu)
 *   0    for int fields     (itmax)
 *   0.0  for lambda         (no regularisation, which is the default)
 *
 * Fields ignored by a given solver are silently disregarded.
 * ------------------------------------------------------------------------- */

typedef struct {
  double atol;    /* NaN  → sqrt(eps(T)) per precision                        */
  double rtol;    /* NaN  → sqrt(eps(T)) per precision                        */
  int    itmax;   /* 0    → solver default                                     */
  int    verbose; /* 0    = silent                                             */
  double lambda;  /* 0.0  = no regularisation (LSQR / LSMR / CGLS / ...)     */
  double tau;     /* NaN  → solver default (TriCG / TriMR : 1.0)              */
  double nu;      /* NaN  → solver default (TriCG / TriMR : -1.0)             */
} KrylovOptions;
""")

  block_section_emitted = false
  for (name, ret, args) in function_sigs
    if startswith(name, "krylov_block_") && !block_section_emitted
      println(io)
      println(io, "/* -------------------------------------------------------------------------")
      println(io, " * Block Krylov interface (block_gmres / block_minres)")
      println(io, " *")
      println(io, " * The right-hand side B and the solution X are m-by-p / n-by-p blocks,")
      println(io, " * stored column-major.  Otherwise the workflow matches the single-vector")
      println(io, " * interface above (create -> solve -> get_X -> free).")
      println(io, " * ------------------------------------------------------------------------- */")
      println(io)
      block_section_emitted = true
    end
    haskey(FUNCTION_DOCS, name) && emit_doc(io, FUNCTION_DOCS[name])
    arg_str = isempty(args) ? "void" : join(["$(ctype) $(aname)" for (aname, ctype) in args], ", ")
    println(io, "$ret $name($arg_str);")
    println(io)
  end

  println(io, """

#ifdef __cplusplus
}
#endif

#endif /* KRYLOV_H */""")
end

println("Generated $out_path")
