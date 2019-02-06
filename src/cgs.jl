# An implementation of CGS for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Iterative methods for sparse linear systems.
# PWS Publishing Company, Boston, USA, 1996.
#
# P. Sonneveld, CGS, A Fast Lanczos-Type Solver for Nonsymmetric Linear systems.
# SIAM Journal on Scientific and Statistical Computing, 10(1), pp. 36--52, 1989.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, October 2018.

export cgs

"""Solve the consistent linear system Ax = b using conjugate gradient squared algorithm.

From "Iterative Methods for Sparse Linear Systems (Y. Saad)" :

«The method is based on a polynomial variant of the conjugate gradients algorithm.
Although related to the so-called bi-conjugate gradients (BCG) algorithm,
it does not involve adjoint matrix-vector multiplications, and the expected convergence
rate is about twice that of the BCG algorithm.

The Conjugate Gradient Squared algorithm works quite well in many cases.
However, one difficulty is that, since the polynomials are squared, rounding errors
tend to be more damaging than in the standard BCG algorithm. In particular, very
high variations of the residual vectors often cause the residual norms computed
to become inaccurate.

TFQMR and BICGSTAB were developed to remedy this difficulty.»

This implementation allows a right preconditioner M.
"""
function cgs(A :: AbstractLinearOperator, b :: AbstractVector{T};
             M :: AbstractLinearOperator=opEye(),
             atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
             itmax :: Int=0, verbose :: Bool=false) where {T <: Number}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("CGS: system of size %d\n", n)

  # Initial solution x₀ and residual r₀.
  x = zeros(T, n) # x₀
  r = copy(b)     # r₀
  # Compute ρ₀ = < r₀,r₀ > and residual norm ‖r₀‖₂.
  ρ = @kdot(n, r, r)
  rNorm = sqrt(ρ)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  u = copy(r)  # u₀
  p = copy(r)  # p₀
  q = zeros(n) # q₋₁

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    y = M * p                 # yₘ = M⁻¹pₘ
    v = A * y                 # vₘ = Ayₘ
    σ = @kdot(n, v, b)        # σₘ = < AM⁻¹pₘ,r₀ >
    α = ρ / σ                 # αₘ = ρₘ / σₘ
    @. q = u - α * v          # qₘ = uₘ - αₘ * AM⁻¹pₘ
    @kaxpy!(n, 1.0, q, u)     # uₘ₊½ = uₘ + qₘ
    z = M * u                 # zₘ = M⁻¹uₘ₊½
    @kaxpy!(n, α, z, x)       # xₘ₊₁ = xₘ + αₘ * M⁻¹(uₘ + qₘ)
    w = A * z                 # wₘ = AM⁻¹(uₘ + qₘ)
    @kaxpy!(n, -α, w, r)      # rₘ₊₁ = rₘ - αₘ * AM⁻¹(uₘ + qₘ)
    ρ_next = @kdot(n, r, b)   # ρₘ₊₁ = < rₘ₊₁,r₀ >
    β = ρ_next / ρ            # βₘ = ρₘ₊₁ / ρₘ
    @. u = r + β * q          # uₘ₊₁ = rₘ₊₁ + βₘ * qₘ
    @kaxpby!(n, 1.0, q, β, p) # pₘ₊₁ = uₘ₊₁ + βₘ * (qₘ + βₘ * pₘ)
    @kaxpby!(n, 1.0, u, β, p)

    # Update ρ.
    ρ = ρ_next # ρₘ ← ρₘ₊₁

    # Update iteration index.
    iter = iter + 1

    # Compute residual norm ‖rₘ‖₂.
    rNorm = @knrm2(n, r)
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
