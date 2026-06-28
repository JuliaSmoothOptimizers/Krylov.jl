module LibKrylov

using LinearAlgebra
using Krylov

include("c_enums.jl")
include("c_operator.jl")
include("c_stores.jl")   # 136 typed stores + dispatch helpers (generated)

# ---------------------------------------------------------------------------
# Function signatures exported to the generated C header.
# Each entry: (c_name, return_type, [(arg_name, c_type), ...])
# This vector is read by scripts/generate_header.jl.
# ---------------------------------------------------------------------------
const function_sigs = Tuple{String, String, Vector{Tuple{String,String}}}[]

macro export_sig(name, ret, args...)
  arg_pairs = [(string(a.args[1]), string(a.args[2])) for a in args]
  push!(function_sigs, (string(name), string(ret), arg_pairs))
  esc(:(nothing))
end

# ---------------------------------------------------------------------------
# krylov_workspace_create
#
# Creates a typed Krylov workspace and writes its opaque pointer into *ws_out.
#
#   solver  : KrylovSolverType enum value (e.g. KRYLOV_CG, KRYLOV_GMRES)
#   m, n    : operator dimensions (m rows, n columns)
#   dtype   : KrylovDataType enum value
#   device  : KrylovDeviceType enum value (currently only KRYLOV_CPU = 0)
#   wopts   : pointer to a KrylovWorkspaceOptions struct, or NULL for defaults
#             (controls memory for GMRES-family solvers and window for MINRES/LS)
#   ws_out  : address of a pointer that receives the workspace handle
#
# Returns 0 on success, -1 on error, -2 on unknown (solver, dtype) combination.
# ---------------------------------------------------------------------------
@export_sig krylov_workspace_create "int" (solver, "KrylovSolverType") (m, "int") (n, "int") (dtype, "KrylovDataType") (device, "KrylovDeviceType") (wopts, "const KrylovWorkspaceOptions*") (ws_out, "void**")

Base.@ccallable function krylov_workspace_create(
    solver   :: Cint,
    m        :: Cint,
    n        :: Cint,
    dtype    :: Cint,
    device   :: Cint,
    wopts_ptr :: Ptr{Cvoid},
    ws_out   :: Ptr{Ptr{Cvoid}},
) :: Cint
  try
    wopts = wopts_ptr == C_NULL ? KrylovWorkspaceOptionsC(Cint(0), Cint(0)) :
                                  unsafe_load(Ptr{KrylovWorkspaceOptionsC}(wopts_ptr))
    _do_create!(solver, m, n, dtype, wopts, ws_out)
  catch e
    @error "krylov_workspace_create" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_default_workspace_options — returns a KrylovWorkspaceOptions with all
# fields set to their "use solver default" sentinel (0).  Call this to
# initialise the struct before overriding memory / window.
# ---------------------------------------------------------------------------
@export_sig krylov_default_workspace_options "KrylovWorkspaceOptions"

Base.@ccallable function krylov_default_workspace_options() :: KrylovWorkspaceOptionsC
    KrylovWorkspaceOptionsC(Cint(0), Cint(0))
end

# ---------------------------------------------------------------------------
# krylov_default_options — returns a KrylovOptions with all fields set to
# their "use solver default" sentinels (NaN for doubles, 0 for ints).
# Always call this to initialise an options struct before overriding fields.
# ---------------------------------------------------------------------------
@export_sig krylov_default_options "KrylovOptions"

Base.@ccallable function krylov_default_options() :: KrylovOptionsC
    KrylovOptionsC(NaN, NaN, Cint(0), Cint(0), 0.0, NaN, NaN)
end

# ---------------------------------------------------------------------------
# krylov_solve
#
#   ws       : workspace handle (from krylov_workspace_create)
#   matvec_A : C callback  void(*)(const void *x, void *y, void *ud)
#              computes y = A*x
#   matvec_At: C callback for y = A'*x, or NULL for CG/GMRES/MINRES/...
#              (required for LSQR, LSMR, CGLS, CRAIG, ...)
#   matvec_M : preconditioner callback (same signature), or NULL
#   b        : right-hand side array of length m
#   c        : second right-hand side (NULL if not needed)
#   userdata : opaque pointer forwarded to every callback
#   opts     : pointer to a KrylovOptions struct, or NULL for all defaults
#
# Returns 0 on success, nonzero on error.
# ---------------------------------------------------------------------------
@export_sig krylov_solve "int" (ws, "void*") (matvec_A, "KrylovMatvec") (matvec_At, "KrylovMatvec") (matvec_M, "KrylovMatvec") (b, "const void*") (c, "const void*") (userdata, "void*") (opts, "const KrylovOptions*")

Base.@ccallable function krylov_solve(
    ws_ptr   :: Ptr{Cvoid},
    fptr_A   :: Ptr{Cvoid},
    fptr_At  :: Ptr{Cvoid},
    fptr_M   :: Ptr{Cvoid},
    b_ptr    :: Ptr{Cvoid},
    c_ptr    :: Ptr{Cvoid},
    userdata :: Ptr{Cvoid},
    opts_ptr :: Ptr{Cvoid},
) :: Cint
  try
    _do_solve!(ws_ptr, fptr_A, fptr_At, fptr_M, b_ptr, c_ptr, userdata, opts_ptr)
  catch e
    @error "krylov_solve" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_get_x — copies the primal solution into a user-provided buffer
# ---------------------------------------------------------------------------
@export_sig krylov_get_x "int" (ws, "void*") (x, "void*") (n, "int")

Base.@ccallable function krylov_get_x(
    ws_ptr :: Ptr{Cvoid},
    x_ptr  :: Ptr{Cvoid},
    n      :: Cint,
) :: Cint
  try
    _do_get_x!(ws_ptr, x_ptr, n)
  catch e
    @error "krylov_get_x" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_get_y — dual solution (TriCG, TriMR, GPMR, BiLQR, TriLQR)
# Returns -2 if the solver has only one solution vector.
# ---------------------------------------------------------------------------
@export_sig krylov_get_y "int" (ws, "void*") (y, "void*") (m, "int")

Base.@ccallable function krylov_get_y(
    ws_ptr :: Ptr{Cvoid},
    y_ptr  :: Ptr{Cvoid},
    m      :: Cint,
) :: Cint
  try
    _do_get_y!(ws_ptr, y_ptr, m)
  catch e
    @error "krylov_get_y" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# Scalar statistics
# ---------------------------------------------------------------------------
@export_sig krylov_is_solved    "int"    (ws, "void*")
@export_sig krylov_niter        "int"    (ws, "void*")
@export_sig krylov_elapsed_time "double" (ws, "void*")

Base.@ccallable function krylov_is_solved(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_is_solved(ws_ptr)
end

Base.@ccallable function krylov_niter(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_niter(ws_ptr)
end

Base.@ccallable function krylov_elapsed_time(ws_ptr :: Ptr{Cvoid}) :: Cdouble
  _do_elapsed_time(ws_ptr)
end

# ---------------------------------------------------------------------------
# krylov_warm_start — sets the initial guess for the next krylov_solve call
# ---------------------------------------------------------------------------
@export_sig krylov_warm_start "int" (ws, "void*") (x0, "const void*") (n, "int")

Base.@ccallable function krylov_warm_start(
    ws_ptr :: Ptr{Cvoid},
    x0_ptr :: Ptr{Cvoid},
    n      :: Cint,
) :: Cint
  try
    _do_warm_start!(ws_ptr, x0_ptr, n)
  catch e
    @error "krylov_warm_start" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_workspace_free — releases the workspace (handle invalid after this)
# Returns 0 on success, 1 if handle was not found.
# ---------------------------------------------------------------------------
@export_sig krylov_workspace_free "int" (ws, "void*")

Base.@ccallable function krylov_workspace_free(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_free!(ws_ptr)
end

# ===========================================================================
# Block Krylov interface (block_gmres / block_minres)
#
# The right-hand side is an m×p block B and the solution an n×p block X, both
# column-major.  The operator is applied to a whole block via a KrylovBlockMatvec
# callback:  void(*)(const void *X, void *Y, int p, void *userdata).
# ===========================================================================

# ---------------------------------------------------------------------------
# krylov_block_workspace_create
#
#   solver : KrylovBlockSolverType (KRYLOV_BLOCK_GMRES, KRYLOV_BLOCK_MINRES)
#   m, n   : operator dimensions
#   p      : block size (number of right-hand sides / columns)
#   dtype  : KrylovDataType
#   device : KrylovDeviceType (KRYLOV_CPU)
#   wopts  : KrylovWorkspaceOptions (memory used by block_gmres), or NULL
#   ws_out : receives the workspace handle
#
# Returns 0 on success, -1 on error, -2 on unknown (solver, dtype) combination.
# ---------------------------------------------------------------------------
@export_sig krylov_block_workspace_create "int" (solver, "KrylovBlockSolverType") (m, "int") (n, "int") (p, "int") (dtype, "KrylovDataType") (device, "KrylovDeviceType") (wopts, "const KrylovWorkspaceOptions*") (ws_out, "void**")

Base.@ccallable function krylov_block_workspace_create(
    solver    :: Cint,
    m         :: Cint,
    n         :: Cint,
    p         :: Cint,
    dtype     :: Cint,
    device    :: Cint,
    wopts_ptr :: Ptr{Cvoid},
    ws_out    :: Ptr{Ptr{Cvoid}},
) :: Cint
  try
    wopts = wopts_ptr == C_NULL ? KrylovWorkspaceOptionsC(Cint(0), Cint(0)) :
                                  unsafe_load(Ptr{KrylovWorkspaceOptionsC}(wopts_ptr))
    _do_block_create!(solver, m, n, p, dtype, wopts, ws_out)
  catch e
    @error "krylov_block_workspace_create" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_block_solve
#
#   ws       : block workspace handle
#   matvec_A : KrylovBlockMatvec computing Y = A*X for a block of p columns
#   matvec_M : preconditioner block matvec (Y = M⁻¹X), or NULL
#   B        : right-hand side block, m×p column-major
#   userdata : forwarded to every callback
#   opts     : KrylovOptions (atol/rtol/itmax/verbose used), or NULL
# ---------------------------------------------------------------------------
@export_sig krylov_block_solve "int" (ws, "void*") (matvec_A, "KrylovBlockMatvec") (matvec_M, "KrylovBlockMatvec") (B, "const void*") (userdata, "void*") (opts, "const KrylovOptions*")

Base.@ccallable function krylov_block_solve(
    ws_ptr   :: Ptr{Cvoid},
    fptr_A   :: Ptr{Cvoid},
    fptr_M   :: Ptr{Cvoid},
    B_ptr    :: Ptr{Cvoid},
    userdata :: Ptr{Cvoid},
    opts_ptr :: Ptr{Cvoid},
) :: Cint
  try
    _do_block_solve!(ws_ptr, fptr_A, fptr_M, B_ptr, userdata, opts_ptr)
  catch e
    @error "krylov_block_solve" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_block_get_X — copies the n×p solution block into a user buffer
# ---------------------------------------------------------------------------
@export_sig krylov_block_get_X "int" (ws, "void*") (X, "void*") (n, "int") (p, "int")

Base.@ccallable function krylov_block_get_X(
    ws_ptr :: Ptr{Cvoid},
    X_ptr  :: Ptr{Cvoid},
    n      :: Cint,
    p      :: Cint,
) :: Cint
  try
    _do_block_get_X!(ws_ptr, X_ptr, n, p)
  catch e
    @error "krylov_block_get_X" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# Block scalar statistics
# ---------------------------------------------------------------------------
@export_sig krylov_block_is_solved    "int"    (ws, "void*")
@export_sig krylov_block_niter        "int"    (ws, "void*")
@export_sig krylov_block_elapsed_time "double" (ws, "void*")

Base.@ccallable function krylov_block_is_solved(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_block_is_solved(ws_ptr)
end

Base.@ccallable function krylov_block_niter(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_block_niter(ws_ptr)
end

Base.@ccallable function krylov_block_elapsed_time(ws_ptr :: Ptr{Cvoid}) :: Cdouble
  _do_block_elapsed_time(ws_ptr)
end

# ---------------------------------------------------------------------------
# krylov_block_warm_start — initial guess (n×p block) for the next solve
# ---------------------------------------------------------------------------
@export_sig krylov_block_warm_start "int" (ws, "void*") (x0, "const void*") (n, "int") (p, "int")

Base.@ccallable function krylov_block_warm_start(
    ws_ptr :: Ptr{Cvoid},
    x0_ptr :: Ptr{Cvoid},
    n      :: Cint,
    p      :: Cint,
) :: Cint
  try
    _do_block_warm_start!(ws_ptr, x0_ptr, n, p)
  catch e
    @error "krylov_block_warm_start" exception=e
    Cint(-1)
  end
end

# ---------------------------------------------------------------------------
# krylov_block_workspace_free — releases a block workspace
# ---------------------------------------------------------------------------
@export_sig krylov_block_workspace_free "int" (ws, "void*")

Base.@ccallable function krylov_block_workspace_free(ws_ptr :: Ptr{Cvoid}) :: Cint
  _do_block_free!(ws_ptr)
end

end  # module LibKrylov
