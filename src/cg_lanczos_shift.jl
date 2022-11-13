# An implementation of the Lanczos version of the conjugate gradient method
# for a family of shifted systems of the form (A + αI) x = b.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
# SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cg_lanczos_shift, cg_lanczos_shift!

"""
    (x, stats) = cg_lanczos_shift(A, b::AbstractVector{FC}, shifts::AbstractVector{T};
                                  M=I, ldiv::Bool=false,
                                  check_curvature::Bool=false, atol::T=√eps(T),
                                  rtol::T=√eps(T), itmax::Int=0,
                                  verbose::Int=0, history::Bool=false,
                                  callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

The Lanczos version of the conjugate gradient method to solve a family
of shifted systems

    (A + αI) x = b  (α = α₁, ..., αₚ)

of size n. The method does _not_ abort if A + αI is not definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be Hermitian and positive definite.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `b`: a vector of length n;
* `shifts`: a vector of length p.

#### Keyword arguments

* `M`:
* `ldiv`:
* `check_curvature`:
* `atol`:
* `rtol`:
* `itmax`:
* `verbose`:
* `history`:
* `callback`:
* `iostream`:

#### Output arguments

* `x`: a vector of p dense vectors, each one of length n;
* `stats`: statistics collected on the run in a [`LanczosShiftStats`](@ref) structure.

#### References

* A. Frommer and P. Maass, [*Fast CG-Based Methods for Tikhonov-Phillips Regularization*](https://doi.org/10.1137/S1064827596313310), SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function cg_lanczos_shift end

function cg_lanczos_shift(A, b :: AbstractVector{FC}, shifts :: AbstractVector{T}; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
  nshifts = length(shifts)
  solver = CgLanczosShiftSolver(A, b, nshifts)
  cg_lanczos_shift!(solver, A, b, shifts; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cg_lanczos!(solver::CgLanczosShiftSolver, A, b, shifts; kwargs...)

where `kwargs` are keyword arguments of [`cg_lanczos_shift`](@ref).

See [`CgLanczosShiftSolver`](@ref) for more details about the `solver`.
"""
function cg_lanczos_shift! end

function cg_lanczos_shift!(solver :: CgLanczosShiftSolver{T,FC,S}, A, b :: AbstractVector{FC}, shifts :: AbstractVector{T};
                           M=I, ldiv :: Bool=false,
                           check_curvature :: Bool=false, atol :: T=√eps(T),
                           rtol :: T=√eps(T), itmax :: Int=0,
                           verbose :: Int=0, history :: Bool=false,
                           callback = solver -> false, iostream :: IO=kstdout) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")

  nshifts = length(shifts)
  (verbose > 0) && @printf(iostream, "CG Lanczos: system of %d equations in %d variables with %d shifts\n", n, n, nshifts)

  # Tests M = Iₙ
  MisI = (M === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :v, S, n)
  Mv, Mv_prev, Mv_next = solver.Mv, solver.Mv_prev, solver.Mv_next
  x, p, σ, δhat = solver.x, solver.p, solver.σ, solver.δhat
  ω, γ, rNorms, converged = solver.ω, solver.γ, solver.rNorms, solver.converged
  not_cv, stats = solver.not_cv, solver.stats
  rNorms_history, indefinite = stats.residuals, stats.indefinite
  reset!(stats)
  v = MisI ? Mv : solver.v

  # Initial state.
  ## Distribute x similarly to shifts.
  for i = 1 : nshifts
    x[i] .= zero(FC)  # x₀
  end
  Mv .= b                             # Mv₁ ← b
  MisI || mulorldiv!(v, M, Mv, ldiv)  # v₁ = M⁻¹ * Mv₁
  β = sqrt(@kdotr(n, v, Mv))          # β₁ = v₁ᴴ M v₁
  rNorms .= β
  if history
    for i = 1 : nshifts
      push!(rNorms_history[i], rNorms[i])
    end
  end

  # Keep track of shifted systems with negative curvature if required.
  indefinite .= false

  if β == 0
    stats.niter = 0
    stats.solved = true
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  # Initialize each p to v.
  for i = 1 : nshifts
    p[i] .= v
  end

  # Initialize Lanczos process.
  # β₁Mv₁ = b
  @kscal!(n, one(FC) / β, v)           # v₁  ←  v₁ / β₁
  MisI || @kscal!(n, one(FC) / β, Mv)  # Mv₁ ← Mv₁ / β₁
  Mv_prev .= Mv

  # Initialize some constants used in recursions below.
  ρ = one(T)
  σ .= β
  δhat .= zero(T)
  ω .= zero(T)
  γ .= one(T)

  # Define stopping tolerance.
  ε = atol + rtol * β

  # Keep track of shifted systems that have converged.
  for i = 1 : nshifts
    converged[i] = rNorms[i] ≤ ε
    not_cv[i] = !converged[i]
  end
  iter = 0
  itmax == 0 && (itmax = 2 * n)

  # Build format strings for printing.
  (verbose > 0) && (fmt = Printf.Format("%5d" * repeat("  %8.1e", nshifts) * "\n"))
  kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms...)

  solved = sum(not_cv) == 0
  tired = iter ≥ itmax
  status = "unknown"
  user_requested_exit = false

  # Main loop.
  while ! (solved || tired || user_requested_exit)
    # Form next Lanczos vector.
    # βₖ₊₁Mvₖ₊₁ = Avₖ - δₖMvₖ - βₖMvₖ₋₁
    mul!(Mv_next, A, v)                  # Mvₖ₊₁ ← Avₖ
    δ = @kdotr(n, v, Mv_next)            # δₖ = vₖᴴ A vₖ
    @kaxpy!(n, -δ, Mv, Mv_next)          # Mvₖ₊₁ ← Mvₖ₊₁ - δₖMvₖ
    if iter > 0
      @kaxpy!(n, -β, Mv_prev, Mv_next)   # Mvₖ₊₁ ← Mvₖ₊₁ - βₖMvₖ₋₁
      @. Mv_prev = Mv                    # Mvₖ₋₁ ← Mvₖ
    end
    @. Mv = Mv_next                      # Mvₖ ← Mvₖ₊₁
    MisI || mulorldiv!(v, M, Mv, ldiv)   # vₖ₊₁ = M⁻¹ * Mvₖ₊₁
    β = sqrt(@kdotr(n, v, Mv))           # βₖ₊₁ = vₖ₊₁ᴴ M vₖ₊₁
    @kscal!(n, one(FC) / β, v)           # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
    MisI || @kscal!(n, one(FC) / β, Mv)  # Mvₖ₊₁ ← Mvₖ₊₁ / βₖ₊₁

    # Check curvature: vₖᴴ(A + sᵢI)vₖ = vₖᴴAvₖ + sᵢ‖vₖ‖² = δₖ + ρₖ * sᵢ with ρₖ = ‖vₖ‖².
    # It is possible to show that σₖ² (δₖ + ρₖ * sᵢ - ωₖ₋₁ / γₖ₋₁) = pₖᴴ (A + sᵢ I) pₖ.
    MisI || (ρ = @kdotr(n, v, v))
    for i = 1 : nshifts
      δhat[i] = δ + ρ * shifts[i]
      γ[i] = 1 / (δhat[i] - ω[i] / γ[i])
    end
    for i = 1 : nshifts
      indefinite[i] |= γ[i] ≤ 0
    end

    # Compute next CG iterate for each shifted system that has not yet converged.
    # Stop iterating on indefinite problems if requested.
    for i = 1 : nshifts
      not_cv[i] = check_curvature ? !(converged[i] || indefinite[i]) : !converged[i]
      if not_cv[i]
        @kaxpy!(n, γ[i], p[i], x[i])
        ω[i] = β * γ[i]
        σ[i] *= -ω[i]
        ω[i] *= ω[i]
        @kaxpby!(n, σ[i], v, ω[i], p[i])

        # Update list of systems that have not converged.
        rNorms[i] = abs(σ[i])
        converged[i] = rNorms[i] ≤ ε
      end
    end

    if length(not_cv) > 0 && history
      for i = 1 : nshifts
        not_cv[i] && push!(rNorms_history[i], rNorms[i])
      end
    end

    # Is there a better way than to update this array twice per iteration?
    for i = 1 : nshifts
      not_cv[i] = check_curvature ? !(converged[i] || indefinite[i]) : !converged[i]
    end
    iter = iter + 1
    kdisplay(iter, verbose) && Printf.format(iostream, fmt, iter, rNorms...)

    user_requested_exit = callback(solver) :: Bool
    solved = sum(not_cv) == 0
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf(iostream, "\n")

  tired               && (status = "maximum number of iterations exceeded")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update stats. TODO: Estimate Anorm and Acond.
  stats.niter = iter
  stats.solved = solved
  stats.status = status
  return solver
end
