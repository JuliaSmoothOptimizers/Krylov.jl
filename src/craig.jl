# An implementation of the Golub-Kahan version of Craig's method
# for the solution of the consistent (under/over-determined or square)
# linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  Ax = b,
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


"""
    (x, y, stats) = craig(A, b; M, N, sqd, λ, atol, btol, rtol, conlim, itmax, verbose, transfer_to_lsqr)

Find the least-norm solution of the consistent linear system

    Ax + λs = b

using the Golub-Kahan implementation of Craig's method, where λ ≥ 0 is a
regularization parameter. This method is equivalent to CGNE but is more
stable.

For a system in the form Ax = b, Craig's method is equivalent to applying
CG to AAᵀy = b and recovering x = Aᵀy. Note that y are the Lagrange
multipliers of the least-norm problem

    minimize ‖x‖  s.t.  Ax = b.

Preconditioners M⁻¹ and N⁻¹ may be provided in the form of linear operators and are
assumed to be symmetric and positive definite.
If `sqd = true`, CRAIG solves the symmetric and quasi-definite system

    [ -N   Aᵀ ] [ x ]   [ 0 ]
    [  A   M  ] [ y ] = [ b ],

which is equivalent to applying CG to `(AN⁻¹Aᵀ + M)y = b` with `Nx = Aᵀy`.

If `sqd = false`, CRAIG solves the symmetric and indefinite system

    [ -N   Aᵀ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

In this case, M⁻¹ can still be specified and indicates the weighted norm in which residuals are measured.

In this implementation, both the x and y-parts of the solution are returned.
"""
function craig(A, b :: AbstractVector{T};
               M=opEye(), N=opEye(), sqd :: Bool=false, λ :: T=zero(T), atol :: T=√eps(T),
               btol :: T=√eps(T), rtol :: T=√eps(T), conlim :: T=1/√eps(T), itmax :: Int=0,
               verbose :: Bool=false, transfer_to_lsqr :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("CRAIG: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₘ and N == Iₙ
  MisI = isa(M, opEye)
  NisI = isa(N, opEye)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Determine the storage type of b
  S = typeof(b)

  x = kzeros(S, n)
  y = kzeros(S, m)

  # When solving a SQD system, set regularization parameter λ = 1.
  sqd && (λ = one(T))

  Mu = copy(b)
  u = M * Mu
  β₁ = sqrt(@kdot(m, u, Mu))
  β₁ == 0 && return x, y, SimpleStats(true, false, [zero(T)], T[], "x = 0 is a zero-residual solution")
  β₁² = β₁^2
  β = β₁
  θ = β₁      # θ will differ from β when there is regularization (λ > 0).
  ξ = -one(T) # Most recent component of x in Range(V).
  δ = λ
  ρ_prev = one(T)

  # Initialize Golub-Kahan process.
  # β₁Mu₁ = b.
  @kscal!(m, one(T)/β₁, u)
  MisI || @kscal!(m, one(T)/β₁, Mu)

  Nv = kzeros(S, n)
  w = kzeros(S, m)  # Used to update y.

  λ > 0 && (w2 = kzeros(S, n))

  Anorm² = zero(T) # Estimate of ‖A‖²_F.
  Anorm  = zero(T)
  Dnorm² = zero(T) # Estimate of ‖(AᵀA)⁻¹‖².
  Acond  = zero(T) # Estimate of cond(A).
  xNorm² = zero(T) # Estimate of ‖x‖².
  xNorm  = zero(T)

  iter = 0
  itmax == 0 && (itmax = m + n)

  rNorm  = β₁
  rNorms = [rNorm;]
  ɛ_c = atol + rtol * rNorm   # Stopping tolerance for consistent systems.
  ɛ_i = atol                  # Stopping tolerance for inconsistent systems.
  ctol = conlim > 0 ? 1/conlim : zero(T)  # Stopping tolerance for ill-conditioned operators.
  verbose && @printf("%5s  %8s  %8s  %8s  %8s  %8s  %7s\n", "Aprod", "‖r‖", "‖x‖", "‖A‖", "κ(A)", "α", "β")
  verbose && @printf("%5d  %8.2e  %8.2e  %8.2e  %8.2e\n", 1, rNorm, xNorm, Anorm, Acond)

  bkwerr = one(T)  # initial value of the backward error ‖r‖ / √(‖b‖² + ‖A‖² ‖x‖²)

  status = "unknown"

  solved_lim = bkwerr ≤ btol
  solved_mach = one(T) + bkwerr ≤ one(T)
  solved_resid_tol = rNorm ≤ ɛ_c
  solved_resid_lim = rNorm ≤ btol + atol * Anorm * xNorm / β₁
  solved = solved_mach | solved_lim | solved_resid_tol | solved_resid_lim

  ill_cond = ill_cond_mach = ill_cond_lim = false

  inconsistent = false
  tired = iter ≥ itmax

  while ! (solved || inconsistent || ill_cond || tired)
    # Generate the next Golub-Kahan vectors
    # 1. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    Aᵀu = Aᵀ * u
    @kaxpby!(n, one(T), Aᵀu, -β, Nv)
    v = N * Nv
    α = sqrt(@kdot(n, v, Nv))
    if α == 0
      inconsistent = true
      continue
    end
    @kscal!(n, one(T)/α, v)
    NisI || @kscal!(n, one(T)/α, Nv)

    Anorm² += α * α + λ * λ

    if λ > 0
      # Givens rotation to zero out the δ in position (k, 2k):
      #      k-1  k   2k     k   2k      k-1  k   2k
      # k   [ θ   α   δ ] [ c₁   s₁ ] = [ θ   ρ      ]
      # k+1 [     β     ] [ s₁  -c₁ ]   [     θ+   γ ]
      (c₁, s₁, ρ) = sym_givens(α, δ)
    else
      ρ = α
    end

    ξ = -θ / ρ * ξ

    if λ > 0
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
    @kaxpby!(m, one(T), u, -θ/ρ_prev, w)  # w = u - θ/ρ_prev * w
    @kaxpy!(m, ξ/ρ, w, y)  # y = y + ξ/ρ * w

    Dnorm² += @knrm2(m, w)

    # 2. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    Av = A * v
    @kaxpby!(m, one(T), Av, -α, Mu)
    u = M * Mu
    β = sqrt(@kdot(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(T)/β, u)
      MisI || @kscal!(m, one(T)/β, Mu)
    end

    # Finish  updates from the first Givens rotation.
    if λ > 0
      θ =  β * c₁
      γ =  β * s₁
    else
      θ = β
    end

    if λ > 0
      # Givens rotation to zero out the γ in position (k+1, 2k)
      #       2k  2k+1     2k  2k+1      2k  2k+1
      # k+1 [  γ    λ ] [ -c₂   s₂ ] = [  0    δ ]
      # k+2 [  0    0 ] [  s₂   c₂ ]   [  0    0 ]
      c₂, s₂, δ = sym_givens(λ, γ)
      @kscal!(n, s₂, w2)
    end

    Anorm² += β * β
    Anorm = sqrt(Anorm²)
    Acond = Anorm * sqrt(Dnorm²)
    xNorm² += ξ * ξ
    xNorm = sqrt(xNorm²)
    rNorm = β * abs(ξ)           # r = - β * ξ * u
    λ > 0 && (rNorm *= abs(c₁))  # r = -c₁ * β * ξ * u when λ > 0.
    push!(rNorms, rNorm)
    iter = iter + 1

    bkwerr = rNorm / sqrt(β₁² + Anorm² * xNorm²)

    ρ_prev = ρ   # Only differs from α if λ > 0.

    verbose && @printf("%5d  %8.2e  %8.2e  %8.2e  %8.2e  %8.1e  %7.1e\n", 1 + 2 * iter, rNorm, xNorm, Anorm, Acond, α, β)

    solved_lim = bkwerr ≤ btol
    solved_mach = one(T) + bkwerr ≤ one(T)
    solved_resid_tol = rNorm ≤ ɛ_c
    solved_resid_lim = rNorm ≤ btol + atol * Anorm * xNorm / β₁
    solved = solved_mach | solved_lim | solved_resid_tol | solved_resid_lim

    ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
    ill_cond_lim = 1 / Acond ≤ ctol
    ill_cond = ill_cond_mach | ill_cond_lim

    inconsistent = false
    tired = iter ≥ itmax
  end

  # transfer to LSQR point if requested
  if λ > 0 && transfer_to_lsqr
    ξ *= -θ / δ
    @kaxpy!(n, ξ, w2, x)
    # TODO: update y
  end

  tired         && (status = "maximum number of iterations exceeded")
  solved        && (status = "solution good enough for the tolerances given")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  inconsistent  && (status = "system may be inconsistent")

  stats = SimpleStats(solved, inconsistent, rNorms, T[], status)
  return (x, y, stats)
end
