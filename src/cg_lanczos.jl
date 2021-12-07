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

export cg_lanczos, cg_lanczos!


"""
    (x, stats) = cg_lanczos(A, b::AbstractVector{FC};
                            M=I, atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                            check_curvature::Bool=false, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

The Lanczos version of the conjugate gradient method to solve the
symmetric linear system

    Ax = b

The method does _not_ abort if A is not definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.

#### References

* A. Frommer and P. Maass, [*Fast CG-Based Methods for Tikhonov-Phillips Regularization*](https://doi.org/10.1137/S1064827596313310), SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function cg_lanczos(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CgLanczosSolver(A, b)
  cg_lanczos!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cg_lanczos!(solver::CgLanczosSolver, args...; kwargs...)

where `args` and `kwargs` are arguments and keyword arguments of [`cg_lanczos`](@ref) without shifts.

See [`CgLanczosSolver`](@ref) for more details about the `solver`.
"""
function cg_lanczos!(solver :: CgLanczosSolver{T,FC,S}, A, b :: AbstractVector{FC};
                     M=I, atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
                     check_curvature :: Bool=false, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("CG Lanczos: system of %d equations in %d variables\n", n, n)

  # Tests M = Iₙ
  MisI = (M === I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :v, S, n)
  x, Mv, Mv_prev = solver.x, solver.Mv, solver.Mv_prev
  p, Mv_next, stats = solver.p, solver.Mv_next, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  v = MisI ? Mv : solver.v

  # Initial state.
  x .= zero(T)              # x₀
  Mv .= b                   # Mv₁ ← b
  MisI || mul!(v, M, Mv)    # v₁ = M⁻¹ * Mv₁
  β = sqrt(@kdot(n, v, Mv)) # β₁ = v₁ᵀ M v₁
  σ = β
  rNorm = σ
  history && push!(rNorms, rNorm)
  if β == 0
    stats.solved = true
    stats.Anorm = zero(T)
    stats.indefinite = false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end
  p .= v

  # Initialize Lanczos process.
  # β₁Mv₁ = b
  @kscal!(n, one(T)/β, v)          # v₁  ←  v₁ / β₁
  MisI || @kscal!(n, one(T)/β, Mv) # Mv₁ ← Mv₁ / β₁
  Mv_prev .= Mv

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  # Initialize some constants used in recursions below.
  ω = zero(T)
  γ = one(T)
  Anorm2 = zero(T)
  β_prev = zero(T)

  # Define stopping tolerance.
  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  indefinite = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  # Main loop.
  while ! (solved || tired || (check_curvature & indefinite))
    # Form next Lanczos vector.
    # βₖ₊₁Mvₖ₊₁ = Avₖ - δₖMvₖ - βₖMvₖ₋₁
    mul!(Mv_next, A, v)      # Mvₖ₊₁ ← Avₖ
    δ = @kdot(n, v, Mv_next) # δₖ = vₖᵀ A vₖ

    # Check curvature. Exit fast if requested.
    # It is possible to show that σₖ² (δₖ - ωₖ₋₁ / γₖ₋₁) = pₖᵀ A pₖ.
    γ = 1 / (δ - ω / γ)      # γₖ = δₖ - ωₖ₋₁ / γₖ₋₁
    indefinite |= (γ ≤ 0)
    (check_curvature & indefinite) && continue

    @kaxpy!(n, -δ, Mv, Mv_next)        # Mvₖ₊₁ ← Mvₖ₊₁ - δₖMvₖ
    if iter > 0
      @kaxpy!(n, -β, Mv_prev, Mv_next) # Mvₖ₊₁ ← Mvₖ₊₁ - βₖMvₖ₋₁
      @. Mv_prev = Mv                  # Mvₖ₋₁ ← Mvₖ
    end
    @. Mv = Mv_next                    # Mvₖ ← Mvₖ₊₁
    MisI || mul!(v, M, Mv)             # vₖ₊₁ = M⁻¹ * Mvₖ₊₁
    β = sqrt(@kdot(n, v, Mv))          # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
    @kscal!(n, one(T)/β, v)            # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
    MisI || @kscal!(n, one(T)/β, Mv)   # Mvₖ₊₁ ← Mvₖ₊₁ / βₖ₊₁
    Anorm2 += β_prev^2 + β^2 + δ^2     # Use ‖Tₖ₊₁‖₂ as increasing approximation of ‖A‖₂.
    β_prev = β

    # Compute next CG iterate.
    @kaxpy!(n, γ, p, x)     # xₖ₊₁ = xₖ + γₖ * pₖ
    ω = β * γ
    σ = -ω * σ              # σₖ₊₁ = - βₖ₊₁ * γₖ * σₖ
    ω = ω * ω               # ωₖ = (βₖ₊₁ * γₖ)²
    @kaxpby!(n, σ, v, ω, p) # pₖ₊₁ = σₖ₊₁ * vₖ₊₁ + ωₖ * pₖ
    rNorm = abs(σ)          # ‖rₖ₊₁‖_M = |σₖ₊₁| because rₖ₊₁ = σₖ₊₁ * vₖ₊₁ and ‖vₖ₊₁‖_M = 1
    history && push!(rNorms, rNorm)
    iter = iter + 1
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (check_curvature & indefinite) ? "negative curvature" : "solution good enough given atol and rtol"

  # Update stats. TODO: Estimate Acond.
  stats.solved = solved
  stats.Anorm = sqrt(Anorm2)
  stats.indefinite = indefinite
  stats.status = status
  return solver
end


"""
    (x, stats) = cg_lanczos(A, b::AbstractVector{FC}, shifts::AbstractVector{FC};
                            M=I, atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                            check_curvature::Bool=false, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

The Lanczos version of the conjugate gradient method to solve a family
of shifted systems

    (A + αI) x = b  (α = α₁, ..., αₙ)

The method does _not_ abort if A + αI is not definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.
"""
function cg_lanczos(A, b :: AbstractVector{FC}, shifts :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  nshifts = length(shifts)
  solver = CgLanczosShiftSolver(A, b, nshifts)
  cg_lanczos!(solver, A, b, shifts; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cg_lanczos!(solver::CgLanczosShiftSolver, args...; kwargs...)

where `args` and `kwargs` are arguments and keyword arguments of [`cg_lanczos`](@ref) with shifts.

See [`CgLanczosShiftSolver`](@ref) for more details about the `solver`.
"""
function cg_lanczos!(solver :: CgLanczosShiftSolver{T,FC,S}, A, b :: AbstractVector{FC}, shifts :: AbstractVector{FC};
                     M=I, atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
                     check_curvature :: Bool=false, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")

  nshifts = length(shifts)
  (verbose > 0) && @printf("CG Lanczos: system of %d equations in %d variables with %d shifts\n", n, n, nshifts)

  # Tests M = Iₙ
  MisI = (M === I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

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
    x[i] .= zero(T)                       # x₀
  end
  Mv .= b                                 # Mv₁ ← b
  MisI || mul!(v, M, Mv)                  # v₁ = M⁻¹ * Mv₁
  β = sqrt(@kdot(n, v, Mv))               # β₁ = v₁ᵀ M v₁
  rNorms .= β
  if history
    for i = 1 : nshifts
      push!(rNorms_history[i], rNorms[i])
    end
  end

  # Keep track of shifted systems with negative curvature if required.
  indefinite .= false

  if β == 0
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
  @kscal!(n, one(T)/β, v)          # v₁  ←  v₁ / β₁
  MisI || @kscal!(n, one(T)/β, Mv) # Mv₁ ← Mv₁ / β₁
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
  if display(iter, verbose)
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n"
    # precompile printf for our particular format
    local_printf(data...) = Core.eval(Main, :(@printf($fmt, $(data)...)))
    local_printf(iter, rNorms...)
  end

  solved = sum(not_cv) == 0
  tired = iter ≥ itmax
  status = "unknown"

  # Main loop.
  while ! (solved || tired)
    # Form next Lanczos vector.
    # βₖ₊₁Mvₖ₊₁ = Avₖ - δₖMvₖ - βₖMvₖ₋₁
    mul!(Mv_next, A, v)                # Mvₖ₊₁ ← Avₖ
    δ = @kdot(n, v, Mv_next)           # δₖ = vₖᵀ A vₖ
    @kaxpy!(n, -δ, Mv, Mv_next)        # Mvₖ₊₁ ← Mvₖ₊₁ - δₖMvₖ
    if iter > 0
      @kaxpy!(n, -β, Mv_prev, Mv_next) # Mvₖ₊₁ ← Mvₖ₊₁ - βₖMvₖ₋₁
      @. Mv_prev = Mv                  # Mvₖ₋₁ ← Mvₖ
    end
    @. Mv = Mv_next                    # Mvₖ ← Mvₖ₊₁
    MisI || mul!(v, M, Mv)             # vₖ₊₁ = M⁻¹ * Mvₖ₊₁
    β = sqrt(@kdot(n, v, Mv))          # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
    @kscal!(n, one(T)/β, v)            # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
    MisI || @kscal!(n, one(T)/β, Mv)   # Mvₖ₊₁ ← Mvₖ₊₁ / βₖ₊₁

    # Check curvature: vₖᵀ(A + sᵢI)vₖ = vₖᵀAvₖ + sᵢ‖vₖ‖² = δₖ + ρₖ * sᵢ with ρₖ = ‖vₖ‖².
    # It is possible to show that σₖ² (δₖ + ρₖ * sᵢ - ωₖ₋₁ / γₖ₋₁) = pₖᵀ (A + sᵢ I) pₖ.
    MisI || (ρ = @kdot(n, v, v))
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
    display(iter, verbose) && local_printf(iter, rNorms...)

    solved = sum(not_cv) == 0
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"

  # Update stats. TODO: Estimate Anorm and Acond.
  stats.solved = solved
  stats.status = status
  return solver
end
