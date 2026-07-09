# An implementation of SQMR for the solution of Hermitian (self-adjoint)
# square linear systems Ax = b.
#
# SQMR is implemented as a wrapper around QMR with centred preconditioning,
# i.e. QMR(A, b, M=M, N=M).  This lets the method accept symmetric
# (including indefinite) preconditioners while reusing the existing QMR
# machinery.  Without a preconditioner, SQMR and MINRES are mathematically
# equivalent.
#
# References:
#
#   R. W. Freund and N. M. Nachtigal,
#   A new Krylov-subspace method for symmetric indefinite linear systems.
#   Proc. 14th IMACS World Congress, pp. 1253--1256, 1994.
#
#   Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.,
#   SIAM, Philadelphia, 2003.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Arnav Kapoor, <arnavkapoor23@iiserb.ac.in>
# Bhopal / Montréal, 2026.

export sqmr, sqmr!

"""
    (x, stats) = sqmr(A, b::AbstractVector{FC};
                      M=I, ldiv::Bool=false,
                      atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = sqmr(A, b, x0::AbstractVector; kwargs...)

SQMR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square Hermitian (self-adjoint) linear system `Ax = b` of size `n`
using the Symmetric Quasi-Minimal Residual (SQMR) method.

SQMR is based on the one-sided symmetric Lanczos process and does not
require products with `Aᴴ`.  The Lanczos vectors are orthogonal with
respect to the M⁻¹-inner product, and the QMR quasi-minimisation is
applied to the resulting symmetric tridiagonal projected system via
Givens rotations.

Unlike MINRES, SQMR supports symmetric *indefinite* preconditioners;
without preconditioning the two methods are mathematically equivalent.

SQMR is implemented as a wrapper around QMR with centred preconditioning
(`M = N`).  See [`qmr`](@ref) for details of the underlying algorithm.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :sqmr`.

For an in-place variant that reuses memory across solves, see [`sqmr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian (possibly indefinite) nonsingular matrix of size `n` used for centred preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* R. W. Freund and N. M. Nachtigal, [*A new Krylov-subspace method for symmetric indefinite linear systems*](https://www.osti.gov/biblio/36034), Proc. 14th IMACS World Congress, pp. 1253--1256, 1994.
* Y. Saad, [*Iterative Methods for Sparse Linear Systems*](https://www-users.cse.umn.edu/~saad/books.html), 2nd edition, SIAM, Philadelphia, 2003.
"""
function sqmr end

"""
    workspace = sqmr!(workspace::SqmrWorkspace, A, b; kwargs...)
    workspace = sqmr!(workspace::SqmrWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`sqmr`](@ref).

See [`SqmrWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :sqmr` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function sqmr! end

def_args_sqmr = (:(A                          ),
                 :(b::AbstractVector{FC}       ))

def_optargs_sqmr = (:(x0::AbstractVector),)

def_kwargs_sqmr = (:(; M          = I                ),
                   :(; ldiv::Bool = false             ),
                   :(; atol::T    = √eps(T)           ),
                   :(; rtol::T    = √eps(T)           ),
                   :(; itmax::Int = 0                 ),
                   :(; timemax::Float64 = Inf         ),
                   :(; verbose::Int     = 0           ),
                   :(; history::Bool    = false       ),
                   :(; callback         = workspace -> false),
                   :(; iostream::IO     = kstdout     ))

def_kwargs_sqmr = extract_parameters.(def_kwargs_sqmr)

args_sqmr    = (:A, :b)
optargs_sqmr = (:x0,)
kwargs_sqmr  = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function sqmr!(workspace :: SqmrWorkspace{T,FC,S}, $(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Build a QmrWorkspace that shares the same storage as the SqmrWorkspace.
    qmr_ws = QmrWorkspace{T,FC,S}(
      workspace.m, workspace.n,
      workspace.uₖ₋₁, workspace.uₖ, workspace.q,
      workspace.vₖ₋₁, workspace.vₖ, workspace.p,
      workspace.Δx, workspace.x, workspace.wₖ₋₂, workspace.wₖ₋₁,
      workspace.t, workspace.s,
      workspace.warm_start, workspace.stats
    )

    # SQMR with centred preconditioning ↔ QMR with M = N.
    qmr!(qmr_ws, A, b, M=M, N=M, ldiv=ldiv;
         atol=atol, rtol=rtol, itmax=itmax, timemax=timemax,
         verbose=verbose, history=history, callback=callback, iostream=iostream)

    # Copy back mutable state.
    workspace.warm_start = qmr_ws.warm_start
    return workspace
  end
end
