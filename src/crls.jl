# An implementation of CRLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the linear system
#
#  AᴴAx = Aᴴb.
#
# This implementation follows the formulation given in
#
# D. C.-L. Fong, Minimum-Residual Methods for Sparse
# Least-Squares using Golubg-Kahan Bidiagonalization,
# Ph.D. Thesis, Stanford University, 2011.
#
# with the difference that it also recurs r = b - Ax.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export crls, crls!

"""
    (x, stats) = crls(A, b::AbstractVector{FC};
                      M=I, ldiv::Bool=false, radius::T=zero(T),
                      λ::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T),
                      itmax::Int=0, timemax::Float64=Inf,
                      verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the linear least-squares problem

    minimize ‖b - Ax‖₂² + λ‖x‖₂²

of size m × n using the Conjugate Residuals (CR) method.
This method is equivalent to applying MINRES to the normal equations

    (AᴴA + λI) x = Aᴴb.

This implementation recurs the residual r := b - Ax.

CRLS produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᴴr‖₂.
It is formally equivalent to LSMR, though can be substantially less accurate,
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

#### Reference

* D. C.-L. Fong, *Minimum-Residual Methods for Sparse, Least-Squares using Golubg-Kahan Bidiagonalization*, Ph.D. Thesis, Stanford University, 2011.
"""
function crls end

"""
    solver = crls!(solver::CrlsSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`crls`](@ref).

See [`CrlsSolver`](@ref) for more details about the `solver`.
"""
function crls! end

def_args_crls = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_crls = (:(; M = I                     ),
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

def_kwargs_crls = extract_parameters.(def_kwargs_crls)

args_crls = (:A, :b)
kwargs_crls = (:M, :ldiv, :radius, :λ, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function crls!(solver :: CrlsSolver{T,FC,S}, $(def_args_crls...); $(def_kwargs_crls...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CRLS: system of %d equations in %d variables\n", m, n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :Ms, S, solver.r)  # The length of Ms is m
    x, p, Ar, q = solver.x, solver.p, solver.Ar, solver.q
    r, Ap, s, stats = solver.r, solver.Ap, solver.s, solver.stats
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)
    Ms  = MisI ? s  : solver.Ms
    Mr  = MisI ? r  : solver.Ms
    MAp = MisI ? Ap : solver.Ms

    kfill!(x, zero(FC))
    kcopy!(m, r, b)      # r ← b
    bNorm = knorm(m, r)  # norm(b - A * x0) if x0 ≠ 0.
    rNorm = bNorm        # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
    history && push!(rNorms, rNorm)
    if bNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      history && push!(ArNorms, zero(T))
      return solver
    end

    MisI || mulorldiv!(Mr, M, r, ldiv)
    mul!(Ar, Aᴴ, Mr)  # - λ * x0 if x0 ≠ 0.
    mul!(s, A, Ar)
    MisI || mulorldiv!(Ms, M, s, ldiv)

    kcopy!(n, p, Ar)  # p ← Ar
    kcopy!(m, Ap, s)  # Ap ← s
    mul!(q, Aᴴ, Ms)   # Ap
    λ > 0 && kaxpy!(n, λ, p, q)  # q = q + λ * p
    γ  = kdotr(m, s, Ms)  # Faster than γ = dot(s, Ms)
    iter = 0
    itmax == 0 && (itmax = m + n)

    ArNorm = knorm(n, Ar)  # Marginally faster than norm(Ar)
    λ > 0 && (γ += λ * ArNorm * ArNorm)
    history && push!(ArNorms, ArNorm)
    ε = atol + rtol * ArNorm
    (verbose > 0) && @printf(iostream, "%5s  %8s  %8s  %5s\n", "k", "‖Aᴴr‖", "‖r‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %8.2e  %8.2e  %.2fs\n", iter, ArNorm, rNorm, start_time |> ktimer)

    status = "unknown"
    on_boundary = false
    solved = ArNorm ≤ ε
    tired = iter ≥ itmax
    psd = false
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || user_requested_exit || overtimed)
      qNorm² = kdotr(n, q, q) # dot(q, q)
      α = γ / qNorm²

      # if a trust-region constraint is give, compute step to the boundary
      # (note that α > 0 in CRLS)
      if radius > 0
        pNorm = knorm(n, p)
        if kdotr(m, Ap, Ap) ≤ ε * sqrt(qNorm²) * pNorm # the quadratic is constant in the direction p
          psd = true # det(AᴴA) = 0
          p = Ar # p = Aᴴr
          pNorm² = ArNorm * ArNorm
          mul!(q, Aᴴ, s)
          α = min(ArNorm^2 / γ, maximum(to_boundary(n, x, p, Ms, radius, flip = false, dNorm2 = pNorm²))) # the quadratic is minimal in the direction Aᴴr for α = ‖Ar‖²/γ
        else
          pNorm² = pNorm * pNorm
          σ = maximum(to_boundary(n, x, p, Ms, radius, flip = false, dNorm2 = pNorm²))
          if α ≥ σ
            α = σ
            on_boundary = true
          end
        end
      end

      kaxpy!(n,  α, p,   x)  # Faster than  x =  x + α *  p
      kaxpy!(n, -α, q,  Ar)  # Faster than Ar = Ar - α *  q
      ArNorm = knorm(n, Ar)
      solved = psd || on_boundary
      solved && continue
      kaxpy!(m, -α, Ap,  r)  # Faster than  r =  r - α * Ap
      mul!(s, A, Ar)
      MisI || mulorldiv!(Ms, M, s, ldiv)
      γ_next = kdotr(m, s, Ms)  # Faster than γ_next = dot(s, s)
      λ > 0 && (γ_next += λ * ArNorm * ArNorm)
      β = γ_next / γ

      kaxpby!(n, one(FC), Ar, β, p)  # Faster than  p = Ar + β *  p
      kaxpby!(m, one(FC), s, β, Ap)  # Faster than Ap =  s + β * Ap
      MisI || mulorldiv!(MAp, M, Ap, ldiv)
      mul!(q, Aᴴ, MAp)
      λ > 0 && kaxpy!(n, λ, p, q)  # q = q + λ * p

      γ = γ_next
      if λ > 0
        rNorm = sqrt(kdotr(m, r, r) + λ * kdotr(n, x, x))
      else
        rNorm = knorm(m, r)
      end
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
    psd                 && (status = "zero-curvature encountered")
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
