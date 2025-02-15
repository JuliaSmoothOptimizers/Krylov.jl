# An implementation of CGLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᴴAx = Aᴴb.
#
# CGLS is formally equivalent to applying the conjugate gradient method
# to the normal equations but should be more stable. It is also formally
# equivalent to LSQR though LSQR should be expected to be more stable on
# ill-conditioned or poorly scaled problems.
#
# This method is described in
#
# M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving linear systems.
# Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
#
# This implementation is the standard formulation, as recommended by
#
# A. Björck, T. Elfving and Z. Strakos, Stability of Conjugate Gradient
# and Lanczos Methods for Linear Least Squares Problems.
# SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cgls, cgls!

"""
    (x, stats) = cgls(A, b::AbstractVector{FC};
                      M=I, ldiv::Bool=false, radius::T=zero(T),
                      λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ‖x‖₂²

of size m × n using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations

    (AᴴA + λI) x = Aᴴb

but is more stable.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖Aᴴr‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. R. Hestenes and E. Stiefel. [*Methods of conjugate gradients for solving linear systems*](https://doi.org/10.6028/jres.049.044), Journal of Research of the National Bureau of Standards, 49(6), pp. 409--436, 1952.
* A. Björck, T. Elfving and Z. Strakos, [*Stability of Conjugate Gradient and Lanczos Methods for Linear Least Squares Problems*](https://doi.org/10.1137/S089547989631202X), SIAM Journal on Matrix Analysis and Applications, 19(3), pp. 720--736, 1998.
"""
function cgls end

"""
    solver = cgls!(solver::CglsSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`cgls`](@ref).

See [`CglsSolver`](@ref) for more details about the `solver`.
"""
function cgls! end

def_args_cgls = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_cgls = (:(; M = I                     ),
                   :(; ldiv::Bool = false        ),
                   :(; radius::T = zero(T)       ),
                   :(; λ::T = zero(T)            ),
                   :(; atol::T = √eps(T)         ),
                   :(; rtol::T = √eps(T)         ),
                   :(; itmax::Int = 0            ),
                   :(; timemax::Float64 = Inf    ),
                   :(; verbose::Int = 0          ),
                   :(; history::Bool = false     ),
                   :(; callback = solver -> false),
                   :(; iostream::IO = kstdout    ))

def_kwargs_cgls = extract_parameters.(def_kwargs_cgls)

args_cgls = (:A, :b)
kwargs_cgls = (:M, :ldiv, :radius, :λ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cgls!(solver :: CglsSolver{T,FC,S}, $(def_args_cgls...); $(def_kwargs_cgls...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CGLS: system of %d equations in %d variables\n", m, n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :Mr, S, solver.r)  # The length of Mr is m
    x, p, s, r, q, stats = solver.x, solver.p, solver.s, solver.r, solver.q, solver.stats
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    Mr = MisI ? r : solver.Mr
    Mq = MisI ? q : solver.Mr

    kfill!(x, zero(FC))
    kcopy!(m, r, b)      # r ← b
    bNorm = knorm(m, r)  # Marginally faster than norm(b)
    if bNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      history && push!(rNorms, zero(T))
      history && push!(ArNorms, zero(T))
      return solver
    end
    MisI || mulorldiv!(Mr, M, r, ldiv)
    mul!(s, Aᴴ, Mr)
    kcopy!(n, p, s)     # p ← s
    γ = kdotr(n, s, s)  # γ = sᴴs
    iter = 0
    itmax == 0 && (itmax = m + n)

    rNorm  = bNorm
    ArNorm = sqrt(γ)
    history && push!(rNorms, rNorm)
    history && push!(ArNorms, ArNorm)
    ε = atol + rtol * ArNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %5s\n", "k", "‖Aᴴr‖", "‖r‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %.2fs\n", iter, ArNorm, rNorm, start_time |> ktimer)

    status = "unknown"
    on_boundary = false
    solved = ArNorm ≤ ε
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || user_requested_exit || overtimed)
      mul!(q, A, p)
      MisI || mulorldiv!(Mq, M, q, ldiv)
      δ = kdotr(m, q, Mq)  # δ = qᴴMq
      λ > 0 && (δ += λ * kdotr(n, p, p))  # δ = δ + pᴴp
      α = γ / δ

      # if a trust-region constraint is give, compute step to the boundary
      σ = radius > 0 ? maximum(to_boundary(n, x, p, Mr, radius)) : α
      if (radius > 0) & (α > σ)
        α = σ
        on_boundary = true
      end

      kaxpy!(n,  α, p, x)  # Faster than x = x + α * p
      kaxpy!(m, -α, q, r)  # Faster than r = r - α * q
      MisI || mulorldiv!(Mr, M, r, ldiv)
      mul!(s, Aᴴ, Mr)
      λ > 0 && kaxpy!(n, -λ, x, s)  # s = A' * r - λ * x
      γ_next = kdotr(n, s, s)  # γ_next = sᴴs
      β = γ_next / γ
      kaxpby!(n, one(FC), s, β, p) # p = s + βp
      γ = γ_next
      rNorm = knorm(m, r)  # Marginally faster than norm(r)
      ArNorm = sqrt(γ)
      history && push!(rNorms, rNorm)
      history && push!(ArNorms, ArNorm)
      iter = iter + 1
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %.2fs\n", iter, ArNorm, rNorm, start_time |> ktimer)
      user_requested_exit = callback(solver) :: Bool
      solved = (ArNorm ≤ ε) || on_boundary
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved              && (status = "solution good enough given atol and rtol")
    on_boundary         && (status = "on trust-region boundary")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
