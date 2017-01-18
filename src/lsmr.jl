# An implementation of LSMR for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the normal equations
#
#  A'Ax = A'b.
#
# LSMR is formally equivalent to applying MINRES to the normal equations
# but should be more stable. It is also formally equivalent to CRLS though
# LSMR should be expected to be more stable on ill-conditioned or poorly
# scaled problems.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# D. C.-L. Fong and M. A. Saunders, LSMR: An Iterative Algorithm for Sparse
# Least Squares Problems, SIAM Journal on Scientific Computing, 33(5),
# pp. 2950--2971, 2011.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, May 2015.

export lsmr

# Methods for various argument types.
include("lsmr_methods.jl")

"""Solve the regularized linear least-squares problem

  minimize ‖b - Ax‖₂² + λ² ‖x‖₂²

using the LSMR method, where λ ≥ 0 is a regularization parameter.
LSQR is formally equivalent to applying MINRES to the normal equations

  (A'A + λ² I) x = A'b

(and therefore to CRLS) but is more stable.

LSMR produces monotonic residuals ‖r‖₂ and optimality residuals ‖A'r‖₂.
rt is formally equivalent to CRLS, though can be slightly more accurate.

Preconditioners M and N may be provided in the form of linear operators and are
assumed to be symmetric and positive definite. If `sqd` is set to `true`,
we solve the symmetric and quasi-definite system

  [ E   A' ] [ r ]   [ b ]
  [ A  -F  ] [ x ] = [ 0 ],

where E = M⁻¹  and F = N⁻¹.

If `sqd` is set to `false` (the default), we solve the symmetric and
indefinite system

  [ E   A' ] [ r ]   [ b ]
  [ A   0  ] [ x ] = [ 0 ].

In this case, `N` can still be specified and indicates the norm
in which `x` should be measured.
"""
function lsmr{T <: Real}(A :: AbstractLinearOperator, b :: Vector{T};
              M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
              sqd :: Bool=false,
              λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
              etol :: Float64=1.0e-8, window :: Int=5,
              itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false)

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("LSMR: system of %d equations in %d variables\n", m, n)

  # If solving an SQD system, set regularization to 1.
  sqd && (λ = 1.0)
  ctol = conlim > 0.0 ? 1/conlim : 0.0
  x = zeros(n)

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu = 1.0*b
  u = M * Mu
  β₁ = sqrt(BLAS.dot(m, u, 1, Mu, 1))
  β₁ == 0.0 && return (x, SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution"))
  β = β₁

  BLAS.scal!(m, 1.0/β₁, u, 1)
  BLAS.scal!(m, 1.0/β₁, Mu, 1)
  Nv = copy(A' * u)
  v = N * Nv
  α = sqrt(BLAS.dot(n, v, 1, Nv, 1))

  ζbar = α * β
  αbar = α
  ρ = 1.0
  ρbar = 1.0
  cbar = 1.0
  sbar = 1.0

  # Initialize variables for estimation of ‖r‖.
  βdd = β
  βd = 0.0
  ρdold = 1.0
  τtildeold = 0.0
  θtilde = 0.0
  ζ = 0.0
  d = 0

  # Initialize variables for estimation of ‖A‖ and cond(A).
  Anorm² = α * α
  maxrbar = 0.0
  minrbar = 1.0e+100;

  # Items for use in stopping rules.
  ctol = conlim > 0.0 ? 1./conlim : 0.0;
  rNorm = β
  rNorms = [rNorm]
  ArNorm = α * β
  ArNorms = [ArNorm]

  xENorm² = 0.0
  err_lbnd = 0.0
  err_vec = zeros(window)

  verbose && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s\n",
                     "Aprod", "‖r‖", "‖A'r‖", "β", "α", "cos", "sin", "‖A‖²")
  verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n",
                     1, β₁, α, β₁, α, 0, 1, Anorm²)

  # A'b = 0 so x = 0 is a minimum least-squares solution
  α == 0.0 && return (x, SimpleStats(true, false, [β₁], [0.0], "x = 0 is a minimum least-squares solution"))
  BLAS.scal!(n, 1.0/α, v, 1)
  BLAS.scal!(n, 1.0/α, Nv, 1)
 
  h = copy(v)
  hbar = zeros(n)

  iter = 0
  itmax == 0 && (itmax = m + n)

  status = "unknown"
  solved = solved_mach = solved_lim = (rNorm <= atol)
  tired  = iter >= itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  zero_resid = zero_resid_mach = zero_resid_lim = false
  fwd_err = false

  while ! (solved || tired || ill_cond)
    iter = iter + 1

    # Generate next Golub-Kahan vectors.
    # 1. βu = Av - αu
    BLAS.scal!(m, -α, Mu, 1)
    BLAS.axpy!(m, 1.0, A * v, 1, Mu, 1)
    u = M * Mu
    β = sqrt(BLAS.dot(m, u, 1, Mu, 1))
    if β != 0.0
      BLAS.scal!(m, 1.0/β, u, 1)
      BLAS.scal!(m, 1.0/β, Mu, 1)

      # 2. αv = A'u - βv
      BLAS.scal!(n, -β, Nv, 1)
      BLAS.axpy!(n, 1.0, A' * u, 1, Nv, 1)
      v = N * Nv
      α = sqrt(BLAS.dot(n, v, 1, Nv, 1))
      if α != 0.0
        BLAS.scal!(n, 1.0/α, v, 1)
        BLAS.scal!(n, 1.0/α, Nv, 1)
      end
    end

    # Continue QR factorization
    (chat, shat, αhat) = sym_givens(αbar, λ)

    ρold = ρ
    (c, s, ρ) = sym_givens(αhat, β)
    θnew = s * α
    αbar = c * α

    ρbarold = ρbar
    ζold = ζ
    θbar = sbar * ρ
    ρtemp = cbar * ρ
    (cbar, sbar, ρbar) = sym_givens(ρtemp, θnew)
    ζ = cbar * ζbar
    ζbar = -sbar * ζbar

    xENorm² = xENorm² + ζ * ζ
    err_vec[mod(iter, window) + 1] = ζ
    iter >= window && (err_lbnd = norm(err_vec))

    # Update h, h_har, x.
    hbar = h - (θbar * ρ / (ρold * ρbarold)) * hbar
    x = x + (ζ / (ρ * ρbar)) * hbar
    h = v - (θnew / ρ) * h

    # Estimate ‖r‖.
    βacute =  chat * βdd
    βcheck = -shat * βdd

    βhat =  c * βacute
    βdd  = -s * βacute

    θtildeold = θtilde
    (ctildeold, stildeold, ρtildeold) = sym_givens(ρdold, θbar)
    θtilde = stildeold * ρbar
    ρdold = ctildeold * ρbar
    βd = -stildeold * βd + ctildeold * βhat

    τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
    τd = (ζ - θtilde * τtildeold) / ρdold
    d = d + βcheck * βcheck
    rNorm = sqrt(d + (βd - τd)^2 + βdd * βdd)
    push!(rNorms, rNorm)

    # Estimate ‖A‖.
    Anorm² += β * β
    Anorm   = sqrt(Anorm²)
    Anorm² += α * α

    # Estimate cond(A).
    maxrbar = max(maxrbar, ρbarold)
    iter > 1 && (minrbar = min(minrbar, ρbarold))
    Acond = max(maxrbar, ρtemp) / min(minrbar, ρtemp)

    # Test for convergence.
    ArNorm = abs(ζbar)
    push!(ArNorms, ArNorm)
    xNorm = BLAS.nrm2(n, x, 1)

    test1 = rNorm / β₁
    test2 = ArNorm / (Anorm * rNorm)
    test3 = 1 / Acond
    t1    = test1 / (1.0 + Anorm * xNorm / β₁)
    rtol  = btol + atol * Anorm * xNorm / β₁

    verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n",
                       1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm²)

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (1.0 + test3 <= 1.0)
    solved_mach = (1.0 + test2 <= 1.0)
    zero_resid_mach = (1.0 + t1 <= 1.0)

    # Stopping conditions based on user-provided tolerances.
    tired  = iter >= itmax
    ill_cond_lim = (test3 <= ctol)
    solved_lim = (test2 <= atol)
    zero_resid_lim = (test1 <= rtol)
    iter >= window && (fwd_err = err_lbnd <= etol * sqrt(xENorm²))

    ill_cond = ill_cond_mach | ill_cond_lim
    solved = solved_mach | solved_lim | zero_resid_mach | zero_resid_lim | fwd_err
  end

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate minimum least-squares solution")
  zero_resid    && (status = "found approximate zero-residual solution")
  fwd_err       && (status = "truncated forward error small enough")

  stats = SimpleStats(solved, !zero_resid, rNorms, ArNorms, status)
  return (x, stats)
end
