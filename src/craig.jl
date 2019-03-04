# An implementation of the Golub-Kahan version of Craig's method
# for the solution of the consistent (under/over-determined or square)
# linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate gradient method
# to the linear system
#
#  AAᵀy = b.
#
# This method, sometimes known under the name CRAIG, is the
# Golub-Kahan implementation of CGNE, and is described in
#
# C. C. Paige and M. A. Saunders, LSQR: An Algorithm for Sparse
# Linear Equations and Sparse Least Squares, ACM Transactions on
# Mathematical Software, Vol 8, No. 1, pp. 43-71, 1982.
#
# and
#
# M. A. Saunders, Solutions of Sparse Rectangular Systems Using
# LSQR and CRAIG, BIT, No. 35, pp. 588-604, 1995.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.
#
# This implementation is strongly inspired from Mike Saunders's.

export craig


"""Find the least-norm solution of the consistent linear system

  Ax + √λs = b

using the Golub-Kahan implementation of Craig's method, where λ ≥ 0 is a
regularization parameter. This method is equivalent to CGNE but is more
stable.

For a system in the form Ax = b, Craig's method is equivalent to applying
CG to AAᵀy = b and recovering x = Aᵀy. Note that y are the Lagrange
multipliers of the least-norm problem

  minimize ‖x‖  subject to Ax = b.

In this implementation, both the x and y-parts of the solution are returned.
"""
function craig(A :: AbstractLinearOperator, b :: AbstractVector{T};
               λ :: Float64=0.0,
               atol :: Float64=1.0e-8, btol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               conlim :: Float64=1.0e+8, itmax :: Int=0,
               verbose :: Bool=false, transfer_to_lsqr :: Bool=false) where T <: Number

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("CRAIG: system of %d equations in %d variables\n", m, n)

  x = zeros(T, n)
  β₁ = @knrm2(m, b)   # Marginally faster than norm(b)
  β₁ == 0 && return x, zeros(m), SimpleStats(true, false, [0.0], T[], "x = 0 is a zero-residual solution")
  β₁² = β₁^2
  β = β₁
  θ = β₁    # θ will differ from β when there is regularization (λ > 0).
  ξ = -1.0  # Most recent component of x in Range(V).
  δ = λ
  ρ_prev = 1.0

  # β₁ u₁ = b.
  u = copy(b)
  @kscal!(m, 1.0/β₁, u)

  v = zeros(T, n)
  w = zeros(T, m)  # Used to update y.

  y = zeros(T, m)
  λ > 0.0 && (w2 = zeros(T, n))

  Anorm² = 0.0   # Estimate of ‖A‖²_F.
  Dnorm² = 0.0   # Estimate of ‖(AᵀA)⁻¹‖².
  Acond  = 0.0   # Estimate of cond(A).
  xNorm² = 0.0   # Estimate of ‖x‖².

  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorm  = β₁
  rNorms = [rNorm;]
  ɛ_c = atol + rtol * rNorm   # Stopping tolerance for consistent systems.
  ɛ_i = atol                  # Stopping tolerance for inconsistent systems.
  ctol = conlim > 0 ? 1/conlim : 0.0  # Stopping tolerance for ill-conditioned operators.
  verbose && @printf("%5s  %8s  %8s  %8s\n", "Aprod", "‖r‖", "‖x‖²", "‖A‖²")
  verbose && @printf("%5d  %8.2e  %8.2e  %8.2e\n", 1, rNorm, xNorm², Anorm²)

  bkwerr = 1.0  # initial value of the backward error ‖r‖ / √(‖b‖² + ‖A‖² ‖x‖²)

  status = "unknown"

  solved_lim = bkwerr ≤ btol
  solved_mach = 1.0 + bkwerr ≤ 1.0
  solved_resid = rNorm ≤ ɛ_c
  solved = solved_mach | solved_lim | solved_resid

  ill_cond = ill_cond_mach = ill_cond_lim = false

  inconsistent = false
  tired = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    # Generate the next Golub-Kahan vectors
    # 1. αv = Aᵀu - βv
    Aᵀu = A.tprod(u)
    @kaxpby!(n, 1.0, Aᵀu, -β, v)
    α = @knrm2(n, v)
    if α == 0.0
      inconsistent = true
      continue
    end
    @kscal!(n, 1.0/α, v)

    Anorm² += α * α + λ * λ

    if λ > 0.0
      # Givens rotation to zero out the δ in position (k, 2k):
      #      k-1  k   2k     k   2k      k-1  k   2k
      # k   [ θ   α   δ ] [ c₁   s₁ ] = [ θ   ρ      ]
      # k+1 [     β     ] [ s₁  -c₁ ]   [     θ+   γ ]
      (c₁, s₁, ρ) = sym_givens(α, δ)
    else
      ρ = α
    end

    ξ = -θ / ρ * ξ

    if λ > 0.0
      # w1 = c₁ * v + s₁ * w2
      # w2 = s₁ * v - c₁ * w2
      # x  = x + ξ * w1
      @kaxpy!(n, ξ * c₁, v, x)
      @kaxpy!(n, ξ * s₁, w2, x)
      @kaxpby!(n, s₁, v, -c₁, w2)
    else
      @kaxpy!(n, ξ, v, x)  # x = x + ξ * v
    end

    # Recur y.
    @kaxpby!(m, 1.0, u, -θ/ρ_prev, w)  # w = u - θ/ρ_prev * w
    @kaxpy!(m, ξ/ρ, w, y)  # y = y + ξ/ρ * w

    Dnorm² += @knrm2(m, w)

    # 2. βu = Av - αu
    Av = A * v
    @kaxpby!(m, 1.0, Av, -α, u)
    β = @knrm2(m, u)
    β > 0.0 && @kscal!(m, 1.0/β, u)

    # Finish  updates from the first Givens rotation.
    if λ > 0.0
      θ =  β * c₁
      γ =  β * s₁
    else
      θ = β
    end

    if λ > 0.0
      # Givens rotation to zero out the γ in position (k+1, 2k)
      #       2k  2k+1    2k  2k+1      2k  2k+1
      # k+1 [  γ    λ ] [ c₂   s₂ ] = [  0    δ ]
      # k+2 [  0    0 ] [ s₂  -c₂ ]   [  0    0 ]
      c₂, s₂, δ = sym_givens(γ, λ)
      @kscal!(n, s₂, w2)
    end

    Anorm² += β * β
    Acond   = sqrt(Anorm²) * sqrt(Dnorm²)
    xNorm² += ξ * ξ
    rNorm   = β * abs(ξ)           # r = -     β * ξ * u
    λ > 0.0 && (rNorm *= abs(c₁))  # r = -c₁ * β * ξ * u when λ > 0.
    push!(rNorms, rNorm)
    iter = iter + 1

    bkwerr = rNorm / sqrt(β₁² + Anorm² * xNorm²)

    ρ_prev = ρ   # Only differs from α if λ > 0.

    verbose && @printf("%5d  %8.2e  %8.2e  %8.2e\n", 1 + 2 * iter, rNorm, xNorm², Anorm²)

    solved_lim = bkwerr ≤ btol
    solved_mach = 1.0 + bkwerr ≤ 1.0
    solved_resid = rNorm ≤ ɛ_c
    solved = solved_mach | solved_lim | solved_resid

    ill_cond_mach = 1 + 1 / Acond ≤ 1
    ill_cond_lim = 1 / Acond ≤ ctol
    ill_cond = ill_cond_mach | ill_cond_lim

    inconsistent = false
    tired = iter ≥ itmax
  end

  inconsistent = !solved_resid  # is there a smarter way?

  # transfer to LSQR point if requested
  if λ > 0 && transfer_to_lsqr
    ξ *= -θ / δ
    @kaxpy!(n, ξ, w2, x)
    # TODO: update y
  end

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  inconsistent  && (status = "system may be inconsistent")
  solved        && (status = "solution good enough for the tolerances given")

  stats = SimpleStats(solved, inconsistent, rNorms, Float64[], status)
  return (x, y, stats)
end
