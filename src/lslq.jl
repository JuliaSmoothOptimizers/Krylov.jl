# An implementation of LSLQ for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the normal equations
#
#  A'Ax = A'b.
#
# LSLQ is formally equivalent to applying SYMMLQ
# to the normal equations but should be more stable.
#
# This implementation accompanies the paper
#
# R. Estrin, D. Orban and M. A. Saunders, LSLQ: An Iterative Method
# for Linear Least-Squares with a Forward Error Minimization Property,
# Cahier du GERAD G-2016-xx, GERAD, Montreal, 2016.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, November 2016.

export lslq

# Methods for various argument types.
include("lslq_methods.jl")

"""Solve the regularized linear least-squares problem

  minimize ‖b - Ax‖₂² + λ² ‖x‖₂²

using the LSLQ method, where λ ≥ 0 is a regularization parameter.
LSLQ is formally equivalent to applying SYMMLQ to the normal equations

  (A'A + λ² I) x = A'b

but is more stable.

LSLQ produces monotonic forward errors ‖x - x*‖₂.

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
function lslq(A :: AbstractLinearOperator, b :: Array{Float64,1}, xsol :: Vector{Float64};
              M :: AbstractLinearOperator=opEye(size(A,1)), N :: AbstractLinearOperator=opEye(size(A,2)),
              sqd :: Bool=false,
              λ :: Float64=0.0, σ :: Float64=0.0,
              atol :: Float64=1.0e-8, btol :: Float64=1.0e-8,
              etol :: Float64=1.0e-8, window :: Int=5,
              utol :: Float64=1.0e-8,
              itmax :: Int=0, conlim :: Float64=1.0e+8, verbose :: Bool=false)

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("LSLQ: system of %d equations in %d variables\n", m, n)

  # If solving an SQD system, set regularization to 1.
  sqd && (λ = 1.0)
  λ² = λ * λ
  σ = sqrt(λ² + σ^2)
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

  Anorm² = α * α

  # condition number estimate
  σmax = 0.0
  σmin = Inf
  Acond  = 0.0

  x_lq = zeros(n)    # LSLQ point
  xlqNorm  = 0.0
  xlqNorm² = 0.0
  x_cg = zeros(n)    # LSQR point
  xcgNorm  = 0.0
  xcgNorm² = 0.0

  w = zeros(n)       # = w₀
  w̄ = copy(v)        # = w̄₁ = v₁

  err_lbnd = 0.0
  err_lbnds = Float64[]
  err_vec = zeros(window)
  err_ubnds_lq = Float64[]
  err_ubnds_cg = Float64[]

  # For paper only
  errs_lq = Float64[]; push!(errs_lq, norm(xsol))
  errs_cg = Float64[]; push!(errs_cg, norm(xsol))

  # Initialize other constants.
  ρ̄ = -σ
  γ̄ = α
  ss = β₁
  c = -1.0
  s = 0.0
  δ = -1.0
  τ = α * β₁
  ζ = 0.0
  csig = -1.0

  rNorm = β₁
  rNorms = [rNorm]
  ArNorm = α * β
  ArNorms = [ArNorm]

  verbose && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s\n",
                     "Aprod", "‖r‖", "‖A'r‖", "β", "α", "cos", "sin", "‖A‖²", "κ(A)", "‖xL‖")
  verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e\n",
                     1, rNorm, ArNorm, β, α, c, s, Anorm², Acond, xlqNorm)

  iter = 0
  itmax == 0 && (itmax = m + n)

  status = "unknown"
  solved = solved_mach = solved_lim = (rNorm ≤ atol)
  tired  = iter ≥ itmax
  ill_cond = ill_cond_mach = ill_cond_lim = false
  zero_resid = zero_resid_mach = zero_resid_lim = false
  fwd_err_lbnd = false
  fwd_err_ubnd = false

  while ! (solved || tired || ill_cond)

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

      # rotate out regularization term if present
      αL = α
      βL = β
      if λ > 0.0
        βL = sqrt(β * β + λ²)
        cL = β / βL
        sL = λ / βL
        αL = cL * α

        # the rotation updates the next regularization parameter
        λ = sqrt(λ² + (sL * α)^2)
      end
      Anorm² = Anorm² + αL * αL + βL * βL;  # = ‖B_{k-1}‖²
      Anorm = sqrt(Anorm²)
    end

    # Continue QR factorization of Bₖ
    # 2. Eliminate β.
    # Q [ Lₖ  β₁ e₁ ] = [ Rₖ   zₖ  ] :
    #   [ β    0    ]   [ 0   ζbar ]
    #
    #       k   k+1     k   k+1      k  k+1
    # k   [ c'   s' ] [ γ̄      ] = [ γ   δ  ]
    # k+1 [ s'  -c' ] [ β   α⁺ ]   [     γ̄ ]
    γ = sqrt(γ̄ * γ̄ + βL * βL)
    τ = -τ * δ / γ  # forward substitution for t
    cp = γ̄ / γ
    sp = βL / γ
    δ = sp * αL
    γ̄ = -cp * αL

    # Continue QR factorization for error estimate
    if σ > 0.0
      μ̄ = -csig * γ
      ρ = sqrt(ρ̄ * ρ̄ + γ * γ)
      csig = ρ̄ / ρ
      ssig = γ / ρ
      ρ̄ = ssig * μ̄ + csig * σ
      μ̄ = -csig * δ

      # determine component of eigenvector and Gauss-Radau parameter
      h = δ * csig / ρ̄
      ω = sqrt(σ * σ - σ * δ * h)

      ρ = sqrt(ρ̄ * ρ̄ + δ * δ)
      csig = ρ̄ / ρ
      ssig = δ / ρ
      ρ̄ = ssig * μ̄ + csig * σ
    end

    # Continue LQ factorization of Rₖ
    ϵ̄ = -γ * c
    η = γ * s
    ϵ = sqrt(ϵ̄ * ϵ̄ + δ * δ)
    c = ϵ̄ / ϵ
    s = δ / ϵ

    # condition number estimate
    # the QLP factorization suggests that the diagonal of M̄ approximates
    # the singular values of B.
    σmax = max(σmax, ϵ, abs(ϵ̄))
    σmin = min(σmin, ϵ, abs(ϵ̄))
    Acond = σmax / σmin

    # forward substitution for z, ζ̄
    ζold = ζ
    ζ = (τ - ζ * η) / ϵ
    ζ̄ = ζ / c

    # residual norm estimate
    rNorm = sqrt((ss * cp - ζ * η)^2 + (ss * sp)^2)
    push!(rNorms, rNorm)

    ArNorm = sqrt((γ * ϵ * ζ)^2 + (δ * η * ζold)^2)
    push!(ArNorms, ArNorm)

    # compute LSQR point
    x_cg = x_lq + ζ̄ * w̄
    xcgNorm² = xlqNorm² + ζ̄ * ζ̄
    push!(errs_cg, norm(xsol - x_cg))

    if σ > 0.0 && iter > 0
      err_ubnd_cg = sqrt(ζ̃ * ζ̃ - ζ̄  * ζ̄ )
      push!(err_ubnds_cg, err_ubnd_cg)
      fwd_err_ubnd = err_ubnd_cg ≤ utol * sqrt(xcgNorm²)
    end

    test1 = rNorm / β₁
    test2 = ArNorm / (Anorm * rNorm)
    test3 = 1 / Acond
    t1    = test1 / (1.0 + Anorm * xlqNorm / β₁)
    rtol  = btol + atol * Anorm * xlqNorm / β₁

    verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e\n",
                       1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm, Acond, xlqNorm)

    # update LSLQ point for next iteration
    w = c * w̄ + s * v
    w̄ = s * w̄ - c * v
    x_lq = x_lq + ζ * w
    xlqNorm² += ζ * ζ
    push!(errs_lq, norm(x - x_lq))

    # check stopping condition based on forward error lower bound
    err_vec[mod(iter, window) + 1] = ζ
    if iter ≥ window
      err_lbnd = norm(err_vec)
      push!(err_lbnds, err_lbnd)
      fwd_err_lbnd = err_lbnd ≤ etol * sqrt(xlqNorm²)
    end

    # compute LQ forward error upper bound
    if σ > 0.0
      η̃ = ω * s
      ϵ̃ = -ω * c
      τ̃ = -τ * δ / ω
      ζ̃ = (τ̃ - ζ * η̃) / ϵ̃
      push!(err_ubnds_lq, abs(ζ̃ ))
    end

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (1.0 + test3 ≤ 1.0)
    solved_mach = (1.0 + test2 ≤ 1.0)
    zero_resid_mach = (1.0 + t1 ≤ 1.0)

    # Stopping conditions based on user-provided tolerances.
    tired  = iter ≥ itmax
    ill_cond_lim = (test3 ≤ ctol)
    solved_lim = (test2 ≤ atol)
    zero_resid_lim = (test1 ≤ rtol)

    ill_cond = ill_cond_mach | ill_cond_lim
    solved = solved_mach | solved_lim | zero_resid_mach | zero_resid_lim | fwd_err_lbnd | fwd_err_ubnd

    iter = iter + 1
  end

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate minimum least-squares solution")
  zero_resid    && (status = "found approximate zero-residual solution")
  fwd_err_lbnd  && (status = "truncated forward error small enough")
  fwd_err_ubnd  && (status = "forward error upper bound small enough")

  stats = SimpleStats(solved, !zero_resid, rNorms, ArNorms, status)
  return (x_lq, x_cg, errs_lq, errs_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats)
end
