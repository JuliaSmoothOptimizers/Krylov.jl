# An implementation of MINRES for the solution of the
# linear system Ax = b, or the linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# where A is square and symmetric.
#
# MINRES is formally equivalent to applying the conjugate residuals method
# to Ax = b when A is positive definite, but is more general and also applies
# to the case where A is indefinite.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Brussels, Belgium, June 2015.
# Montreal, August 2015.

export minres, minres!


"""
    (x, stats) = minres(A, b::AbstractVector{T};
                        M=I, λ::T=zero(T), atol::T=√eps(T)/100,
                        rtol::T=√eps(T)/100, ratol :: T=zero(T), 
                        rrtol :: T=zero(T), etol::T=√eps(T),
                        window::Int=5, itmax::Int=0,
                        conlim::T=1/√eps(T), restart::Bool=false,
                        verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the shifted linear least-squares problem

    minimize ‖b - (A + λI)x‖₂²

or the shifted linear system

    (A + λI) x = b

using the MINRES method, where λ ≥ 0 is a shift parameter,
where A is square and symmetric.

MINRES is formally equivalent to applying CR to Ax=b when A is positive
definite, but is typically more stable and also applies to the case where
A is indefinite.

MINRES produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᵀr‖₂.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be symmetric and positive definite.

#### Reference

* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function minres(A, b :: AbstractVector{T}; window :: Int=5, kwargs...) where T <: AbstractFloat
  solver = MinresSolver(A, b, window=window)
  minres!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function minres!(solver :: MinresSolver{T,S}, A, b :: AbstractVector{T};
                 M=I, λ :: T=zero(T), atol :: T=√eps(T)/100, rtol :: T=√eps(T)/100, 
                 ratol :: T=zero(T), rrtol :: T=zero(T), etol :: T=√eps(T),
                 itmax :: Int=0, conlim :: T=1/√eps(T), restart :: Bool=false,
                 verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("MINRES: system of size %d\n", n)

  # Tests M == Iₙ
  MisI = (M == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")

  # Set up workspace.
  allocate_if(!MisI  , solver, :v , S, n)
  allocate_if(restart, solver, :Δx, S, n)
  Δx, x, r1, r2, w1, w2, y = solver.Δx, solver.x, solver.r1, solver.r2, solver.w1, solver.w2, solver.y
  err_vec, stats = solver.err_vec, solver.stats
  rNorms, ArNorms, Aconds = stats.residuals, stats.Aresiduals, stats.Acond
  reset!(stats)
  v = MisI ? r2 : solver.v

  ϵM = eps(T)
  ctol = conlim > 0 ? 1 / conlim : zero(T)

  # Initial solution x₀
  restart && (Δx .= x)
  x .= zero(T)

  if restart
    mul!(r1, A, Δx)
    (λ ≠ 0) && @kaxpy!(n, λ, Δx, r1)
    @kaxpby!(n, one(T), b, -one(T), r1)
  else
    r1 .= b
  end

  # Initialize Lanczos process.
  # β₁ M v₁ = b.
  r2 .= r1
  MisI || mul!(v, M, r1)
  β₁ = @kdot(m, r1, v)
  β₁ < 0 && error("Preconditioner is not positive definite")
  if β₁ == 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    history && push!(rNorms, β₁)
    history && push!(ArNorms, zero(T))
    history && push!(Aconds, zero(T))
    return solver
  end
  β₁ = sqrt(β₁)
  β = β₁

  oldβ = zero(T)
  δbar = zero(T)
  ϵ = zero(T)
  rNorm = β₁
  history && push!(rNorms, β₁)
  ϕbar = β₁
  rhs1 = β₁
  rhs2 = zero(T)
  γmax = zero(T)
  γmin = T(Inf)
  cs = -one(T)
  sn = zero(T)
  w1 .= zero(T)
  w2 .= zero(T)

  ANorm² = zero(T)
  ANorm = zero(T)
  Acond = zero(T)
  history && push!(Aconds, Acond)
  ArNorm = zero(T)
  history && push!(ArNorms, ArNorm)
  xNorm = zero(T)

  xENorm² = zero(T)
  err_lbnd = zero(T)
  window = length(err_vec)
  err_vec .= zero(T)

  iter = 0
  itmax == 0 && (itmax = 2*n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s  %7s\n", "Aprod", "‖r‖", "‖Aᵀr‖", "β", "cos", "sin", "‖A‖", "κ(A)", "test1", "test2")
  display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n", 0, rNorm, ArNorm, β, cs, sn, ANorm, Acond)

  tol = atol + rtol * β₁
  rNormtol = ratol + rrtol * β₁ 
  stats.status = "unknown"
  solved = solved_mach = solved_lim = (rNorm ≤ rtol)
  tired  = iter ≥ itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  zero_resid = zero_resid_mach = zero_resid_lim = (rNorm ≤ tol)
  fwd_err = false

  while !(solved || tired || ill_cond)
    iter = iter + 1

    # Generate next Lanczos vector.
    mul!(y, A, v)
    λ ≠ 0 && @kaxpy!(n, λ, v, y)             # (y = y + λ * v)
    @kscal!(n, one(T) / β, y)
    iter ≥ 2 && @kaxpy!(n, -β / oldβ, r1, y) # (y = y - β / oldβ * r1)

    α = @kdot(n, v, y) / β
    @kaxpy!(n, -α / β, r2, y)  # y = y - α / β * r2

    # Compute w.
    δ = cs * δbar + sn * α
    if iter == 1
      w = w2
    else
      iter ≥ 3 && @kscal!(n, -ϵ, w1)
      w = w1
      @kaxpy!(n, -δ, w2, w)
    end
    @kaxpy!(n, one(T) / β, v, w)

    @. r1 = r2
    @. r2 = y
    MisI || mul!(v, M, r2)
    oldβ = β
    β = @kdot(n, r2, v)
    β < 0 && error("Preconditioner is not positive definite")
    β = sqrt(β)
    ANorm² = ANorm² + α * α + oldβ * oldβ + β * β

    # Apply rotation to obtain
    #  [ δₖ    ϵₖ₊₁    ] = [ cs  sn ] [ δbarₖ  0    ]
    #  [ γbar  δbarₖ₊₁ ]   [ sn -cs ] [ αₖ     βₖ₊₁ ]
    γbar = sn * δbar - cs * α
    ϵ = sn * β
    δbar = -cs * β
    root = sqrt(γbar * γbar + δbar * δbar)
    ArNorm = ϕbar * root  # = ‖Aᵀrₖ₋₁‖
    history && push!(ArNorms, ArNorm)

    # Compute the next plane rotation.
    γ = sqrt(γbar * γbar + β * β)
    γ = max(γ, ϵM)
    cs = γbar / γ
    sn = β / γ
    ϕ = cs * ϕbar
    ϕbar = sn * ϕbar

    # Final update of w.
    @kscal!(n, one(T) / γ, w)

    # Update x.
    @kaxpy!(n, ϕ, w, x)  # x = x + ϕ * w
    xENorm² = xENorm² + ϕ * ϕ

    # Update directions for x.
    if iter ≥ 2
      @kswap(w1, w2)
    end

    # Compute lower bound on forward error.
    err_vec[mod(iter, window) + 1] = ϕ
    iter ≥ window && (err_lbnd = norm(err_vec))

    γmax = max(γmax, γ)
    γmin = min(γmin, γ)
    ζ = rhs1 / γ
    rhs1 = rhs2 - δ * ζ
    rhs2 = -ϵ * ζ

    # Estimate various norms.
    ANorm = sqrt(ANorm²)
    xNorm = @knrm2(n, x)
    ϵA = ANorm * ϵM
    ϵx = ANorm * xNorm * ϵM
    ϵr = ANorm * xNorm * rtol
    d = γbar
    d == 0 && (d = ϵA)

    rNorm = ϕbar

    test1 = rNorm / (ANorm * xNorm)
    test2 = root / ANorm
    history && push!(rNorms, rNorm)

    Acond = γmax / γmin
    history && push!(Aconds, Acond)

    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", iter, rNorm, ArNorm, β, cs, sn, ANorm, Acond, test1, test2)

    if iter == 1 && β / β₁ ≤ 10 * ϵM
      # Aᵀb = 0 so x = 0 is a minimum least-squares solution
      stats.solved, stats.inconsistent = true, true
      stats.status = "x is a minimum least-squares solution"
      return solver
    end

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (one(T) + one(T) / Acond ≤ one(T))
    solved_mach = (one(T) + test2 ≤ one(T))
    zero_resid_mach = (one(T) + test1 ≤ one(T))
    resid_decrease_mach = (rNorm + one(T) ≤ one(T))
    # solved_mach = (ϵx ≥ β₁)

    # Stopping conditions based on user-provided tolerances.
    tired = iter ≥ itmax
    ill_cond_lim = (one(T) / Acond ≤ ctol)
    solved_lim = (test2 ≤ tol)
    zero_resid_lim = (test1 ≤ tol)
    resid_decrease_lim = (rNorm ≤ rNormtol)
    iter ≥ window && (fwd_err = err_lbnd ≤ etol * sqrt(xENorm²))

    zero_resid = zero_resid_mach | zero_resid_lim
    resid_decrease = resid_decrease_mach | resid_decrease_lim
    ill_cond = ill_cond_mach | ill_cond_lim
    solved = solved_mach | solved_lim | zero_resid | fwd_err | resid_decrease
  end
  (verbose > 0) && @printf("\n")

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate minimum least-squares solution")
  zero_resid    && (status = "found approximate zero-residual solution")
  fwd_err       && (status = "truncated forward error small enough")

  # Update x
  restart && @kaxpy!(n, one(T), Δx, x)

  # Update stats
  stats.solved = solved
  stats.inconsistent = !zero_resid
  stats.status = status
  return solver
end
