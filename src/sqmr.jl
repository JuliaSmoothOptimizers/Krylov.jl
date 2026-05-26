# An implementation of SQMR for the solution of symmetric square linear systems Ax = b.
#
# SQMR is implemented here as a symmetric wrapper around QMR with centered
# preconditioning. This lets the method accept symmetric indefinite
# preconditioners while reusing the existing QMR machinery.

export sqmr, sqmr!

"""
    (x, stats) = sqmr(A, b::AbstractVector{FC};
                      M=I, ldiv::Bool=false, atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = sqmr(A, b, x0::AbstractVector; kwargs...)

SQMR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the square symmetric linear system `Ax = b` of size `n` using SQMR.

SQMR is a centered variant of QMR specialized to symmetric systems. It supports
symmetric preconditioners that are not necessarily positive definite.
The method requires support for `adjoint(M)` when a preconditioner is provided.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :sqmr`.

For an in-place variant that reuses memory across solves, see [`sqmr!`](@ref).

#### Input arguments

* `A`: a linear operator that models a symmetric matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a symmetric nonsingular matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.
"""
function sqmr end

"""
    workspace = sqmr!(workspace::QmrWorkspace, A, b; kwargs...)
    workspace = sqmr!(workspace::QmrWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`sqmr`](@ref).

See [`QmrWorkspace`](@ref) for instructions on how to create the `workspace`.
"""
function sqmr! end

def_args_sqmr = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_optargs_sqmr = (:(x0::AbstractVector),)

def_kwargs_sqmr = (:(; M = I                        ),
                  :(; ldiv::Bool = false           ),
                  :(; atol::T = √eps(T)            ),
                  :(; rtol::T = √eps(T)            ),
                  :(; itmax::Int = 0               ),
                  :(; timemax::Float64 = Inf       ),
                  :(; verbose::Int = 0             ),
                  :(; history::Bool = false        ),
                  :(; callback = workspace -> false),
                  :(; iostream::IO = kstdout       ))

def_kwargs_sqmr = extract_parameters.(def_kwargs_sqmr)

args_sqmr = (:A, :b)
optargs_sqmr = (:x0,)
kwargs_sqmr = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function sqmr!(workspace :: QmrWorkspace{T,FC,S}, $(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
    return qmr!(workspace, A, b; c = b, M = M, N = M, ldiv = ldiv, atol = atol, rtol = rtol,
                itmax = itmax, timemax = timemax, verbose = verbose, history = history,
                callback = callback, iostream = iostream)
  end

  function sqmr($(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    return qmr(A, b; c = b, M = M, N = M, ldiv = ldiv, atol = atol, rtol = rtol,
               itmax = itmax, timemax = timemax, verbose = verbose, history = history,
               callback = callback, iostream = iostream)
  end

  function sqmr($(def_args_sqmr...), $(def_optargs_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    return qmr(A, b, x0; c = b, M = M, N = M, ldiv = ldiv, atol = atol, rtol = rtol,
               itmax = itmax, timemax = timemax, verbose = verbose, history = history,
               callback = callback, iostream = iostream)
  end

  krylov_solve(::Val{:sqmr}, $(def_args_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} =
    sqmr($(args_sqmr...); $(kwargs_sqmr...))

  krylov_solve(::Val{:sqmr}, $(def_args_sqmr...), $(def_optargs_sqmr...); $(def_kwargs_sqmr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}} =
    sqmr($(args_sqmr...), $(optargs_sqmr...); $(kwargs_sqmr...))
end