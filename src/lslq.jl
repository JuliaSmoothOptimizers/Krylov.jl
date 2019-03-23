# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, November 2016-January 2017.

export lslq


"""
    lslq(A, b, λ=0.0)

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ² ‖x‖₂²

using the LSLQ method, where λ ≥ 0 is a regularization parameter.
LSLQ is formally equivalent to applying SYMMLQ to the normal equations

    (AᵀA + λ² I) x = Aᵀb

but is more stable.

#### Main features

* the solution estimate is updated along orthogonal directions
* the norm of the solution estimate ‖xᴸₖ‖₂ is increasing
* the error ‖eₖ‖₂ := ‖xᴸₖ - x*‖₂ is decreasing
* it is possible to transition cheaply from the LSLQ iterate to the LSQR iterate if there is an advantage (there always is in terms of error)
* if `A` is rank deficient, identify the minimum least-squares solution

#### Input arguments

* `A::AbstractLinearOperator`
* `b::Vector{Float64}`

#### Optional arguments

* `M::AbstractLinearOperator=opEye(size(A,1))`: a symmetric and positive definite dual preconditioner
* `N::AbstractLinearOperator=opEye(size(A,2))`: a symmetric and positive definite primal preconditioner
* `sqd::Bool=false` indicates whether or not we are solving a symmetric and quasi-definite augmented system
  If `sqd = true`, we solve the symmetric and quasi-definite system

      [ E   Aᵀ ] [ r ]   [ b ]
      [ A  -F  ] [ x ] = [ 0 ],

  where E = M⁻¹  and F = N⁻¹.

  If `sqd = false`, we solve the symmetric and indefinite system

      [ E   Aᵀ ] [ r ]   [ b ]
      [ A   0  ] [ x ] = [ 0 ].

  In this case, `N` can still be specified and indicates the norm in which `x` and the forward error should be measured.
* `λ::Float64=0.0` is a regularization parameter (see the problem statement above)
* `σ::Float64=0.0` is an underestimate of the smallest nonzero singular value of `A`---setting `σ` too large will result in an error in the course of the iterations
* `atol::Float64=1.0e-8` is a stopping tolerance based on the residual
* `btol::Float64=1.0e-8` is a stopping tolerance used to detect zero-residual problems
* `etol::Float64=1.0e-8` is a stopping tolerance based on the lower bound on the error
* `window::Int=5` is the number of iterations used to accumulate a lower bound on the error
* `utol::Float64=1.0e-8` is a stopping tolerance based on the upper bound on the error
* `itmax::Int=0` is the maximum number of iterations (0 means no imposed limit)
* `conlim::Float64=1.0e+8` is the limit on the estimated condition number of `A` beyond which the solution will be abandoned
* `verbose::Bool=false` determines verbosity.

#### Return values

`lslq()` returns the tuple `(x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats)` where

* `x_lq::Vector{Float64}` is the LQ solution estimate
* `x_cg::Vector{Float64}` is the CG solution estimate (i.e., the LSQR point)
* `err_lbnds::Vector{Float64}` is a vector of lower bounds on the LQ error---the vector is empty if `window` is set to zero
* `err_ubnds_lq::Vector{Float64}` is a vector of upper bounds on the LQ error---the vector is empty if `σ == 0` is left at zero
* `err_ubnds_cg::Vector{Float64}` is a vector of upper bounds on the CG error---the vector is empty if `σ == 0` is left at zero
* `stats::SimpleStats` collects other statistics on the run.

#### Stopping conditions

The iterations stop as soon as one of the following conditions holds true:

* the optimality residual is sufficiently small (`stats.status = "found approximate minimum least-squares solution"`) in the sense that either
  * ‖Aᵀr‖ / (‖A‖ ‖r‖) ≤ atol, or
  * 1 + ‖Aᵀr‖ / (‖A‖ ‖r‖) ≤ 1
* an approximate zero-residual solution has been found (`stats.status = "found approximate zero-residual solution"`) in the sense that either
  * ‖r‖ / ‖b‖ ≤ btol + atol ‖A‖ * ‖xᴸ‖ / ‖b‖, or
  * 1 + ‖r‖ / ‖b‖ ≤ 1
* the estimated condition number of `A` is too large in the sense that either
  * 1/cond(A) ≤ 1/conlim (`stats.status = "condition number exceeds tolerance"`), or
  * 1 + 1/cond(A) ≤ 1 (`stats.status = "condition number seems too large for this machine"`)
* the lower bound on the LQ forward error is less than etol * ‖xᴸ‖
* the upper bound on the CG forward error is less than utol * ‖xᶜ‖

#### References

* R. Estrin, D. Orban and M. A. Saunders, *Estimates of the 2-Norm Forward Error for SYMMLQ and CG*, Cahier du GERAD G-2016-70, GERAD, Montreal, 2016. DOI http://dx.doi.org/10.13140/RG.2.2.19581.77288.
* R. Estrin, D. Orban and M. A. Saunders, *LSLQ: An Iterative Method for Linear Least-Squares with an Error Minimization Property*, Cahier du GERAD G-2017-xx, GERAD, Montreal, 2017.
"""
function lslq(A :: AbstractLinearOperator, b :: AbstractVector{T};
              M :: AbstractLinearOperator=opEye(),
              N :: AbstractLinearOperator=opEye(),
              sqd :: Bool=false, λ :: Float64=0.0, σ :: Float64=0.0,
              atol :: Float64=1.0e-8, btol :: Float64=1.0e-8, etol :: Float64=1.0e-8,
              window :: Int=5, utol :: Float64=1.0e-8, itmax :: Int=0,
              conlim :: Float64=1.0e+8, verbose :: Bool=false) where T <: Number

  m, n = size(A)
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("LSLQ: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ and N == Iₘ
  MisI = isa(M, opEye)
  NisI = isa(N, opEye)

  # If solving an SQD system, set regularization to 1.
  sqd && (λ = 1.0)
  λ² = λ * λ
  ctol = conlim > 0.0 ? 1/conlim : 0.0

  x_lq = zeros(T, n)    # LSLQ point
  err_lbnds = T[]
  err_ubnds_lq = T[]
  err_ubnds_cg = T[]

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu = copy(b)
  u = M * Mu
  β₁ = sqrt(@kdot(m, u, Mu))
  β₁ == 0.0 && return (x_lq, zeros(T, n), err_lbnds, err_ubnds_lq, err_ubnds_cg,
                       SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution"))
  β = β₁

  @kscal!(m, 1.0/β₁, u)
  MisI || @kscal!(m, 1.0/β₁, Mu)
  Aᵀu = A.tprod(u)
  Nv = copy(Aᵀu)
  v = N * Nv
  α = sqrt(@kdot(n, v, Nv))  # = α₁

  # Aᵀb = 0 so x = 0 is a minimum least-squares solution
  α == 0.0 && return (x_lq, zeros(T, n), err_lbnds, err_ubnds_lq, err_ubnds_cg,
                      SimpleStats(true, false, [β₁], [0.0], "x = 0 is a minimum least-squares solution"))
  @kscal!(n, 1.0/α, v)
  NisI || @kscal!(n, 1.0/α, Nv)

  Anorm = α
  Anorm² = α * α

  # condition number estimate
  σmax = 0.0
  σmin = Inf
  Acond  = 0.0

  xlqNorm  = 0.0
  xlqNorm² = 0.0
  xcgNorm  = 0.0
  xcgNorm² = 0.0

  w̄ = copy(v) # w̄₁ = v₁

  err_lbnd = 0.0
  err_vec = zeros(T, window)

  # Initialize other constants.
  αL = α
  βL = β
  ρ̄ = -σ
  γ̄ = α
  ss = β₁
  c = -1.0
  s = 0.0
  δ = -1.0
  τ = α * β₁
  ζ = 0.0
  ζ̄  = 0.0
  ζ̃  = T(0)
  csig = -1.0

  rNorm = β₁
  rNorms = [rNorm]
  ArNorm = α * β
  ArNorms = [ArNorm]

  verbose && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s\n",
                     "Aprod", "‖r‖", "‖Aᵀr‖", "β", "α", "cos", "sin", "‖A‖²", "κ(A)", "‖xL‖")
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
    # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    Av = A * v
    @kaxpby!(m, 1.0, Av, -α, Mu)
    u = M * Mu
    β = sqrt(@kdot(m, u, Mu))
    if β != 0.0
      @kscal!(m, 1.0/β, u)
      MisI || @kscal!(m, 1.0/β, Mu)

      # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
      Aᵀu = A.tprod(u)
      @kaxpby!(n, 1.0, Aᵀu, -β, Nv)
      v = N * Nv
      α = sqrt(@kdot(n, v, Nv))
      if α != 0.0
        @kscal!(n, 1.0/α, v)
        NisI || @kscal!(n, 1.0/α, Nv)
      end

      # rotate out regularization term if present
      αL = α
      βL = β
      if λ != 0.0
        (cL, sL, βL) = sym_givens(β, λ)
        αL = cL * α

        # the rotation updates the next regularization parameter
        λ = sqrt(λ² + (sL * α)^2)
      end
      Anorm² = Anorm² + αL * αL + βL * βL;  # = ‖Lₖ‖²
      Anorm = sqrt(Anorm²)
    end

    # Continue QR factorization of Bₖ
    #
    #       k   k+1     k   k+1      k  k+1
    # k   [ c'   s' ] [ γ̄      ] = [ γ   δ  ]
    # k+1 [ s'  -c' ] [ β   α⁺ ]   [     γ̄ ]
    (cp, sp, γ) = sym_givens(γ̄, βL)
    τ = -τ * δ / γ  # forward substitution for t
    δ = sp * αL
    γ̄ = -cp * αL

    if σ > 0.0
      # Continue QR factorization for error estimate
      μ̄ = -csig * γ
      (csig, ssig, ρ) = sym_givens(ρ̄, γ)
      ρ̄ = ssig * μ̄ + csig * σ
      μ̄ = -csig * δ

      # determine component of eigenvector and Gauss-Radau parameter
      h = δ * csig / ρ̄
      ω = sqrt(σ * (σ - δ * h))
      (csig, ssig, ρ) = sym_givens(ρ̄, δ)
      ρ̄ = ssig * μ̄ + csig * σ
    end

    # Continue LQ factorization of Rₖ
    ϵ̄ = -γ * c
    η = γ * s
    (c, s, ϵ) = sym_givens(ϵ̄, δ)

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

    # Compute ‖x_cg‖₂
    xcgNorm² = xlqNorm² + ζ̄ * ζ̄

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
    @kaxpy!(n, c * ζ, w̄, x_lq)
    @kaxpy!(n, s * ζ, v, x_lq)

    # compute w̄
    @kaxpby!(n, -c, v, s, w̄)

    xlqNorm² += ζ * ζ
    xlqNorm = sqrt(xlqNorm²)

    # check stopping condition based on forward error lower bound
    err_vec[mod(iter, window) + 1] = ζ
    if iter ≥ window
      err_lbnd = norm(err_vec)
      push!(err_lbnds, err_lbnd)
      fwd_err_lbnd = err_lbnd ≤ etol * xlqNorm
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

  # compute LSQR point
  @kaxpby!(n, 1.0, x_lq, ζ̄ , w̄)
  x_cg = w̄

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate minimum least-squares solution")
  zero_resid    && (status = "found approximate zero-residual solution")
  fwd_err_lbnd  && (status = "forward error lower bound small enough")
  fwd_err_ubnd  && (status = "forward error upper bound small enough")

  stats = SimpleStats(solved, !zero_resid, rNorms, ArNorms, status)
  return (x_lq, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats)
end
