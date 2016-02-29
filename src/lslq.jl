# An implementation of LSLQ for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the normal equations
#
#  A'Ax = A'b.
#
# LSLQ is formally equivalent to applying the SYMMLQ method
# to the normal equations but should be more stable.
#
# This implementation follows the original implementation by
# Dominique Orban and Michael Saunders described in
#
# D. Orban and M. A. Saunders, LSLQ: An Iterative Method for Linear
# Least Squares with a Forward Error Minimization Property,
# in preparation, July 2015.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, October 2015.

export lslq

# Methods for various argument types.
include("lslq_methods.jl")

"""Solve the regularized linear least-squares problem

  minimize ‖b - Ax‖₂² + λ² ‖x‖₂²

using the LSLQ method, where λ ≥ 0 is a regularization parameter.
LSLQ is formally equivalent to applying SYMMLQ to the normal equations

  (A'A + λ² I) x = A'b

but is more stable.

LSLQ produces monotonic solution norms ‖x‖₂ and forward errors ‖x - x*‖
but not residuals ‖r‖₂ or optimality residuals ‖A'r‖₂.

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
function lslq(A :: AbstractLinearOperator, b :: Array{Float64,1}, x_exact :: Vector{Float64};
              M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
              sqd :: Bool=false,
              λ :: Float64=0.0, atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
              etol :: Float64=1.0e-8, window :: Int=5,
              itmax :: Int=0, conlim :: Float64=1.0e+8, 
              a :: Float64=0.0, verbose :: Bool=false)

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("LSLQ: system of %d equations in %d variables\n", m, n)

  # If solving an SQD system, set regularization to 1.
  sqd && (λ = 1.0)
  λ² = λ * λ
  ctol = conlim > 0.0 ? 1/conlim : 0.0

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu = copy(b)
  u = M * Mu
  β₁ = sqrt(BLAS.dot(m, u, 1, Mu, 1))
  β₁ == 0.0 && return (x, SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution"))
  β = β₁

  BLAS.scal!(m, 1.0/β₁, u, 1)
  BLAS.scal!(m, 1.0/β₁, Mu, 1)
  Nv = copy(A' * u)
  v = N * Nv
  α = sqrt(BLAS.dot(n, v, 1, Nv, 1))

  # A'b = 0 so x = 0 is a minimum least-squares solution
  α == 0.0 && return (x, SimpleStats(true, false, [β₁], [0.0], "x = 0 is a minimum least-squares solution"))

  BLAS.scal!(n, 1.0/α, v, 1)
  BLAS.scal!(n, 1.0/α, Nv, 1)

  β̄ = α * β          # = β̄₁
  xlqNorm  = 0.0
  xlqNorm² = 0.0

  # We need the next u before entering the main loop.
  # βu = Av - αu
  BLAS.scal!(m, -α, Mu, 1)
  BLAS.axpy!(m, 1.0, A * v, 1, Mu, 1)
  u = M * Mu
  β = sqrt(BLAS.dot(m, u, 1, Mu, 1))
  if β != 0.0
    BLAS.scal!(m, 1.0/β, u, 1)
    BLAS.scal!(m, 1.0/β, Mu, 1)
  end

  # Initialize other constants.
  ᾱ = α * α + β * β + λ²  # = ᾱ₁
  γ̄ = ᾱ                   # = γ̄₁
  ζ = 0
  ζ̄ = β̄ / γ̄
  s = 0.0
  c = -1.0

  if (a > 0)
    μ = sqrt(ᾱ )         # = μ₁
    ρ = sqrt(μ * μ - a)  # = ρ₁
  end

  w = zeros(n)       # = w₀
  x_lq = zeros(n)    # = x₀
  fwdErrs_lq = [norm(x_exact)]

  w̄ = copy(v)        # = w̄₁ = v₁
  x_cg = ζ̄ * w̄
  fwdErrs_cg = [norm(x_exact - x_cg)]
  xcgNorm  = ζ̄
  xcgNorm² = ζ̄ * ζ̄
  lc = β̄  # Used to update the residual of the normal equations at the CG point.

  err_lbnd = 0.0
  err_lbnds = Float64[]
  err_ubnds = Float64[]
  err_ubnds_cg = Float64[]
  err_vec = zeros(window)

  rNorm = β₁
  rNorms = [rNorm]
  Anorm² = ᾱ
  Anorm  = sqrt(Anorm²)
  Acond  = 1.0
  ArNorm = α
  ArNorms = [ArNorm]
  ArNorms_cg = Float64[]

  xsol_est = Float64[]

  verbose && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s  %7s\n",
                     "Aprod", "‖r‖", "‖A'r‖", "β", "α", "cos", "sin", "‖A‖", "‖x‖")
  verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n",
                     1, β₁, α, β, α, c, s, Anorm, xlqNorm)

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
    # αv = A'u - βv
    BLAS.scal!(n, -β, Nv, 1)
    BLAS.axpy!(n, 1.0, A' * u, 1, Nv, 1)
    v = N * Nv
    α = sqrt(BLAS.dot(n, v, 1, Nv, 1))
    if α != 0.0
      BLAS.scal!(n, 1.0/α, v, 1)
      BLAS.scal!(n, 1.0/α, Nv, 1)
    end
    β̄ = α * β  # this is α₂ β₂ = β̄₂ at the first pass through the loop.

    # βu = Av - αu
    BLAS.scal!(m, -α, Mu, 1)
    BLAS.axpy!(m, 1.0, A * v, 1, Mu, 1)
    u = M * Mu
    β = sqrt(BLAS.dot(m, u, 1, Mu, 1))
    if β != 0.0
      BLAS.scal!(m, 1.0/β, u, 1)
      BLAS.scal!(m, 1.0/β, Mu, 1)
    end
    ᾱ = α * α + β * β + λ²  # this is ᾱ₂ at the first pass through the loop.

    if (a > 0)
      ν = β̄  / μ                 # ν₂ at first pass of loop 
      ω = a + (μ * μ * ν * ν / (ρ * ρ)) # ω₂ at first pass of loop

      σ = μ * ν / ρ              # σ₂ at first pass of loop
      μ = sqrt(ᾱ  - ν * ν)       # μ₂ at first pass of loop
      ρ = sqrt(μ * μ + ν * ν - a - σ * σ) # ρ₂ at first pass of loop
    end
    Anorm² = Anorm² + ᾱ  # = ‖Bₖ₋₁‖²

    # Continue LQ factorization
    ϵ = s * β̄
    δ̄ = -c * β̄

    # Eliminate β̄.
    #       k-2  k-1  k      k-1  k  k+1     k-2  k-1  k
    # k-1 [ δ₋   γ̄₋   β̄  ] [ c    s     ]   [ δ₋  γ₋      ]
    # k   [ ϵ    δ̄    ᾱ  ] [ s   -c     ] = [ ϵ   δ    γ̄  ]
    # k+1 [           β̄⁺ ] [         1  ]   [     ϵ⁺   δ̄⁺ ]
    # (c, s, γ) = sym_givens(γ̄, β̄)
    γ = sqrt(γ̄^2 + β̄^2);
    c = γ̄ / γ; s = β̄ / γ
    δ = c * δ̄ + s * ᾱ
    γ̄ = s * δ̄ - c * ᾱ
    ζ_old = ζ
    ζ = c * ζ̄
    ζ̄ = -(δ * ζ + ϵ * ζ_old) / γ̄

    if (a > 0)

      ψ = c * δ̄ + s * ω
      ω̄ = s * δ̄ - c * ω

      ζ_norm = -(ψ * ζ + ϵ * ζ_old) / ω̄
    end

    if (a > 0)
      xsolNorm² = xlqNorm² + ζ_norm * ζ_norm
      push!(xsol_est, xsolNorm²)
      push!(err_ubnds, xsolNorm² - xlqNorm²)
      push!(err_ubnds_cg, xsolNorm² - xlqNorm² - ζ̄  * ζ̄ )
    end

    xlqNorm² = xlqNorm² + ζ * ζ
    xcgNorm² = xcgNorm² + ζ̄ * ζ̄
    err_vec[mod(iter, window) + 1] = ζ
    if iter >= window
      err_lbnd = norm(err_vec)
      push!(err_lbnds, err_lbnd)
    end

    w = c * w̄ + s * v
    w̄ = s * w̄ - c * v
    # BLAS.scal!(n, s, w̄, 1)
    # BLAS.axpy!(n, -c, v, 1, w̄, 1);     # w̄ = -c * v + s * w̄
    # BLAS.axpy!(n,  ζ, w, 1, x_lq, 1);  # xlq = xlq + ζ * w
    x_lq = x_lq + ζ * w
    x_cg = x_lq + ζ̄ * w̄ 

    push!(fwdErrs_lq, norm(x_lq - x_exact))
    push!(fwdErrs_cg, norm(x_cg - x_exact))

    Anorm = sqrt(Anorm²)
    Acond = 1.0  #Anorm * sqrt(dNorm²)

    ArNorm = sqrt(γ * γ * ζ * ζ + ϵ * ϵ * ζ_old * ζ_old)
    push!(ArNorms, ArNorm)

    ArNorm_cg = lc * s / c
    push!(ArNorms_cg, ArNorm_cg)
    lc = lc * s

    # TODO: Estimate rNorm.
    rNorm = norm(b - A * x_lq)
    push!(rNorms, rNorm)

    xlqNorm = sqrt(xlqNorm²)
    test1 = rNorm / β₁
    test2 = ArNorm / (Anorm * rNorm)
    test3 = 1 / Acond
    t1    = test1 / (1.0 + Anorm * xlqNorm / β₁)
    rtol  = btol + atol * Anorm * xlqNorm / β₁

    verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e\n",
                       1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm, xlqNorm)

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
    iter >= window && (fwd_err = err_lbnd <= etol * xlqNorm)

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
  return (x_lq, x_cg, fwdErrs_lq, fwdErrs_cg, err_lbnds, err_ubnds, err_ubnds_cg, xsol_est, stats)
end
