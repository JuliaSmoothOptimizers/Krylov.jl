# An implementation of TFQMR for the solution of the square linear system Ax = b.
#
# This method is described in
#
# R. W. Freund, A Transpose-Free Quasi-Minimal Residual Method for Non-Hermitian Linear Systems,
# SIAM Journal on Scientific Computing, 14(2), pp. 470--482, 1993.
#
# C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations,
# SIAM Frontiers in Applied Mathematics, 16, pp. 575--601, 1995.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, October 2018.

export tfqmr

"""Solve the consistent linear system Ax = b using the transpose-free quasi-minimal residual algorithm.
TFMQR may be used to solve unsymmetric systems.

This implementation allows a right preconditioner M.
"""
function tfqmr(A :: AbstractLinearOperator, b :: AbstractVector{T};
               M :: AbstractLinearOperator=opEye(),
               atol :: T=√eps(T), rtol :: T=√eps(T),
               itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  m == n || error("System must be square")
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("TFQMR: system of size %d\n", n)

  # Initial solution x₀.
  x = zeros(T, n) # x₀
  # Compute ρ₀ = < r₀,r₀ > and residual norm ‖r₀‖₂.
  ρ = @kdot(n, b, b)
  rNorm = sqrt(ρ)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  y = copy(b)     # y₀
  w = copy(b)     # w₀
  d = zeros(T, n) # d₀
  τ = rNorm       # τ₀
  η = zero(T)     # η₀
  s = zero(T)     # s₀
  c = one(T)      # c₀
  z = M * y       # z₀
  u = A * z       # u₀
  v = copy(u)     # v₀

  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    σ = @kdot(n, v, b)                             # σₖ = < vₖ,r₀ >
    α = ρ / σ                                      # αₖ = ρₖ / σₖ

    @kaxpy!(n, -α, u, w)                           # w₂ₖ₊₁ = w₂ₖ - αₖ * u₂ₖ
    @kaxpby!(n, one(T), z, s^2 / c^2 * η / α, d)   # d₂ₖ₊₁ = z₂ₖ + η₂ₖ * (s₂ₖ / c₂ₖ)² / αₖ * d₂ₖ
    nw = @knrm2(n, w)                              # [c₂ₖ₊₁  s₂ₖ₊₁][   τ₂ₖ  ] = [γ₂ₖ]
    (c, s, γ) = sym_givens(τ, nw)                  # [s₂ₖ₊₁ -c₂ₖ₊₁][‖w₂ₖ₊₁‖₂] = [ 0 ]
    τ = s * τ                                      # τ₂ₖ₊₁ = s₂ₖ₊₁ * τ₂ₖ
    η = c * c * α                                  # η₂ₖ₊₁ = (c₂ₖ₊₁)² * αₖ
    @kaxpy!(n, η, d, x)                            # x₂ₖ₊₁ = x₂ₖ + η₂ₖ₊₁ * d₂ₖ₊₁
    @kaxpy!(n, -α, v, y)                           # y₂ₖ₊₁ = y₂ₖ - αₖ * vₖ
    z = M * y                                      # z₂ₖ₊₁ = M⁻¹ * y₂ₖ
    u = A * z                                      # u₂ₖ₊₁ = A * z₂ₖ₊₁

    @kaxpy!(n, -α, u, w)                           # w₂ₖ₊₂ = w₂ₖ₊₁ - αₖ * u₂ₖ₊₁
    @kaxpby!(n, one(T), z, s^2 / c^2 * η / α, d)   # d₂ₖ₊₂ = z₂ₖ₊₁ + η₂ₖ₊₁ * (s₂ₖ₊₁ / c₂ₖ₊₁)² / αₖ * d₂ₖ₊₁
    nw = @knrm2(n, w)                              # [c₂ₖ₊₂  s₂ₖ₊₂][  τ₂ₖ₊₁ ] = [γ₂ₖ₊₁]
    (c, s, γ) = sym_givens(τ, nw)                  # [s₂ₖ₊₂ -c₂ₖ₊₂][‖w₂ₖ₊₂‖₂] = [  0  ]
    τ = s * τ                                      # τ₂ₖ₊₂ = s₂ₖ₊₂ * τ₂ₖ₊₁
    η = c * c * α                                  # η₂ₖ₊₂ = (c₂ₖ₊₂)² * αₖ
    @kaxpy!(n, η, d, x)                            # x₂ₖ₊₂ = x₂ₖ₊₁ + η₂ₖ₊₂ * d₂ₖ₊₂

    ρ_next = @kdot(n, w, b)                        # ρₖ₊₁ = < w₂ₖ₊₂,r₀ >
    β = ρ_next / ρ                                 # βₖ = ρₖ₊₁ / ρₖ
    ρ = ρ_next                                     # ρₖ ← ρₖ₊₁

    @kaxpby!(n, one(T), w, β, y)                   # y₂ₖ₊₂ = w₂ₖ₊₂ + βₖ * y₂ₖ₊₁
    @kaxpby!(n, one(T), u, β, v); @kscal!(n, β, v) # vₐᵤₓ = βₖ * (u₂ₖ₊₁ + βₖ * vₖ)
    z = M * y                                      # z₂ₖ₊₂ = M⁻¹ * y₂ₖ₊₂
    u = A * z                                      # u₂ₖ₊₂ = A * z₂ₖ₊₂

    @kaxpy!(n, one(T), u, v)                       # vₖ₊₁ = u₂ₖ₊₂ + vₐᵤₓ
    
    # Update iteration index.
    iter = iter + 1
    
    # Compute residual norm estimate, ‖rₖ‖₂ ≤ √(2k+1) * τ₂ₖ.
    rNorm = sqrt(2 * iter + 1) * τ
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
