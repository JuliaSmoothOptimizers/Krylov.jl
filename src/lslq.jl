# An implementation of LSLQ for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᵀAx = Aᵀb.
#
# LSLQ is formally equivalent to applying SYMMLQ to the normal equations
# but should be more stable.
#
# This method is described in
#
# R. Estrin, D. Orban and M.A. Saunders
# LSLQ: An Iterative Method for Linear Least-Squares with an Error Minimization Property
# SIAM Journal on Matrix Analysis and Applications, 40(1), pp. 254--275, 2019.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, November 2016-January 2017.

export lslq, lslq!


"""
    (x, stats) = lslq(A, b::AbstractVector{FC};
                      M=I, N=I, sqd::Bool=false, λ::T=zero(T),
                      atol::T=√eps(T), btol::T=√eps(T), etol::T=√eps(T),
                      window::Int=5, utol::T=√eps(T), itmax::Int=0,
                      σ::T=zero(T), transfer_to_lsqr::Bool=false, 
                      conlim::T=1/√eps(T), verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ²‖x‖₂²

using the LSLQ method, where λ ≥ 0 is a regularization parameter.
LSLQ is formally equivalent to applying SYMMLQ to the normal equations

    (AᵀA + λ²I) x = Aᵀb

but is more stable.

#### Main features

* the solution estimate is updated along orthogonal directions
* the norm of the solution estimate ‖xᴸₖ‖₂ is increasing
* the error ‖eₖ‖₂ := ‖xᴸₖ - x*‖₂ is decreasing
* it is possible to transition cheaply from the LSLQ iterate to the LSQR iterate if there is an advantage (there always is in terms of error)
* if `A` is rank deficient, identify the minimum least-squares solution

#### Optional arguments

* `M`: a symmetric and positive definite dual preconditioner
* `N`: a symmetric and positive definite primal preconditioner
* `sqd` indicates whether or not we are solving a symmetric and quasi-definite augmented system

If `sqd = true`, we solve the symmetric and quasi-definite system

    [ E    A ] [ r ]   [ b ]
    [ Aᵀ  -F ] [ x ] = [ 0 ],

where E and F are symmetric and positive definite.
The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹ + ‖x‖²_F.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᵀKx`.
LSLQ is then equivalent to applying SYMMLQ to `(AᵀE⁻¹A + F)x = AᵀE⁻¹b` with `r = E⁻¹(b - Ax)`.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.

If `sqd` is set to `false` (the default), we solve the symmetric and
indefinite system

    [ E    A ] [ r ]   [ b ]
    [ Aᵀ   0 ] [ x ] = [ 0 ].

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹.

In this case, `N` can still be specified and indicates the weighted norm in which `x` and `Aᵀr` should be measured.
`r` can be recovered by computing `E⁻¹(b - Ax)`.

* `λ` is a regularization parameter (see the problem statement above)
* `σ` is an underestimate of the smallest nonzero singular value of `A`---setting `σ` too large will result in an error in the course of the iterations
* `atol` is a stopping tolerance based on the residual
* `btol` is a stopping tolerance used to detect zero-residual problems
* `etol` is a stopping tolerance based on the lower bound on the error
* `window` is the number of iterations used to accumulate a lower bound on the error
* `utol` is a stopping tolerance based on the upper bound on the error
* `transfer_to_lsqr` return the CG solution estimate (i.e., the LSQR point) instead of the LQ estimate
* `itmax` is the maximum number of iterations (0 means no imposed limit)
* `conlim` is the limit on the estimated condition number of `A` beyond which the solution will be abandoned
* `verbose` determines verbosity.

#### Return values

`lslq` returns the tuple `(x, stats)` where

* `x` is the LQ solution estimate
* `stats` collects other statistics on the run in a LSLQStats

* `stats.err_lbnds` is a vector of lower bounds on the LQ error---the vector is empty if `window` is set to zero
* `stats.err_ubnds_lq` is a vector of upper bounds on the LQ error---the vector is empty if `σ == 0` is left at zero
* `stats.err_ubnds_cg` is a vector of upper bounds on the CG error---the vector is empty if `σ == 0` is left at zero
* `stats.error_with_bnd` is a boolean indicating whether there was an error in the upper bounds computation (cancellation errors, too large σ ...)

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

* R. Estrin, D. Orban and M. A. Saunders, [*Euclidean-norm error bounds for SYMMLQ and CG*](https://doi.org/10.1137/16M1094816), SIAM Journal on Matrix Analysis and Applications, 40(1), pp. 235--253, 2019.
* R. Estrin, D. Orban and M. A. Saunders, [*LSLQ: An Iterative Method for Linear Least-Squares with an Error Minimization Property*](https://doi.org/10.1137/17M1113552), SIAM Journal on Matrix Analysis and Applications, 40(1), pp. 254--275, 2019.
"""
function lslq(A, b :: AbstractVector{FC}; window :: Int=5, kwargs...) where FC <: FloatOrComplex
  solver = LslqSolver(A, b, window=window)
  lslq!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function lslq!(solver :: LslqSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, N=I, sqd :: Bool=false, λ :: T=zero(T),
               atol :: T=√eps(T), btol :: T=√eps(T), etol :: T=√eps(T),
               utol :: T=√eps(T), itmax :: Int=0, σ :: T=zero(T),
               transfer_to_lsqr :: Bool=false, conlim :: T=1/√eps(T),
               verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("LSLQ: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₙ and N == Iₘ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")
  NisI || (promote_type(eltype(N), T) == T) || error("eltype(N) can't be promoted to $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  x, Nv, Aᵀu, w̄ = solver.x, solver.Nv, solver.Aᵀu, solver.w̄
  Mu, Av, err_vec, stats = solver.Mu, solver.Av, solver.err_vec, solver.stats
  rNorms, ArNorms, err_lbnds = stats.residuals, stats.Aresiduals, stats.err_lbnds
  err_ubnds_lq, err_ubnds_cg = stats.err_ubnds_lq, stats.err_ubnds_cg
  reset!(stats)
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  # If solving an SQD system, set regularization to 1.
  sqd && (λ = one(T))
  λ² = λ * λ
  ctol = conlim > 0 ? 1/conlim : zero(T)

  x .= zero(T)  # LSLQ point

  # Initialize Golub-Kahan process.
  # β₁ M u₁ = b.
  Mu .= b
  MisI || mul!(u, M, Mu)
  β₁ = sqrt(@kdot(m, u, Mu))
  if β₁ == 0
    stats.solved, stats.inconsistent = true, false
    stats.error_with_bnd = false
    history && push!(rNorms, zero(T))
    history && push!(ArNorms, zero(T))
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end
  β = β₁

  @kscal!(m, one(T)/β₁, u)
  MisI || @kscal!(m, one(T)/β₁, Mu)
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)
  α = sqrt(@kdot(n, v, Nv))  # = α₁

  # Aᵀb = 0 so x = 0 is a minimum least-squares solution
  if α == 0
    stats.solved, stats.inconsistent = true, false
    stats.error_with_bnd = false
    history && push!(rNorms, β₁)
    history && push!(ArNorms, zero(T))
    stats.status = "x = 0 is a minimum least-squares solution"
    return solver
  end
  @kscal!(n, one(T)/α, v)
  NisI || @kscal!(n, one(T)/α, Nv)

  Anorm = α
  Anorm² = α * α

  # condition number estimate
  σmax = zero(T)
  σmin = Inf
  Acond  = zero(T)

  xlqNorm  = zero(T)
  xlqNorm² = zero(T)
  xcgNorm  = zero(T)
  xcgNorm² = zero(T)

  w̄ .= v  # w̄₁ = v₁

  err_lbnd = zero(T)
  window = length(err_vec)
  err_vec .= zero(T)
  complex_error_bnd = false

  # Initialize other constants.
  αL = α
  βL = β
  ρ̄ = -σ
  γ̄ = α
  ss = β₁
  c = -one(T)
  s = zero(T)
  δ = -one(T)
  τ = α * β₁
  ζ = zero(T)
  ζ̄  = zero(T)
  ζ̃  = zero(T)
  csig = -one(T)

  rNorm = β₁
  history && push!(rNorms, rNorm)
  ArNorm = α * β
  history && push!(ArNorms, ArNorm)

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s\n", "Aprod", "‖r‖", "‖Aᵀr‖", "β", "α", "cos", "sin", "‖A‖²", "κ(A)", "‖xL‖")
  display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e\n", 1, rNorm, ArNorm, β, α, c, s, Anorm², Acond, xlqNorm)

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
    mul!(Av, A, v)
    @kaxpby!(m, one(T), Av, -α, Mu)
    MisI || mul!(u, M, Mu)
    β = sqrt(@kdot(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(T)/β, u)
      MisI || @kscal!(m, one(T)/β, Mu)

      # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
      mul!(Aᵀu, Aᵀ, u)
      @kaxpby!(n, one(T), Aᵀu, -β, Nv)
      NisI || mul!(v, N, Nv)
      α = sqrt(@kdot(n, v, Nv))
      if α ≠ 0
        @kscal!(n, one(T)/α, v)
        NisI || @kscal!(n, one(T)/α, Nv)
      end

      # rotate out regularization term if present
      αL = α
      βL = β
      if λ ≠ 0
        (cL, sL, βL) = sym_givens(β, λ)
        αL = cL * α

        # the rotation updates the next regularization parameter
        λ = sqrt(λ² + (sL * α)^2)
      end
      Anorm² = Anorm² + αL * αL + βL * βL  # = ‖Lₖ‖²
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

    if σ > 0 && !complex_error_bnd
      # Continue QR factorization for error estimate
      μ̄ = -csig * γ
      (csig, ssig, ρ) = sym_givens(ρ̄, γ)
      ρ̄ = ssig * μ̄ + csig * σ
      μ̄ = -csig * δ

      # determine component of eigenvector and Gauss-Radau parameter
      h = δ * csig / ρ̄
      disc = σ * (σ - δ * h)
      disc < 0 ? complex_error_bnd = true : ω = sqrt(disc)
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
    history && push!(rNorms, rNorm)

    ArNorm = sqrt((γ * ϵ * ζ)^2 + (δ * η * ζold)^2)
    history && push!(ArNorms, ArNorm)

    # Compute ‖x_cg‖₂
    xcgNorm² = xlqNorm² + ζ̄ * ζ̄

    if σ > 0 && iter > 0 && !complex_error_bnd
      disc = ζ̃ * ζ̃ - ζ̄  * ζ̄ 
      if disc < 0
        complex_error_bnd = true
      else
        err_ubnd_cg = sqrt(disc)
        history && push!(err_ubnds_cg, err_ubnd_cg)
        fwd_err_ubnd = err_ubnd_cg ≤ utol * sqrt(xcgNorm²)
      end
    end

    test1 = rNorm / β₁
    test2 = ArNorm / (Anorm * rNorm)
    test3 = 1 / Acond
    t1    = test1 / (one(T) + Anorm * xlqNorm / β₁)
    rtol  = btol + atol * Anorm * xlqNorm / β₁

    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e\n", 1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm, Acond, xlqNorm)

    # update LSLQ point for next iteration
    @kaxpy!(n, c * ζ, w̄, x)
    @kaxpy!(n, s * ζ, v, x)

    # compute w̄
    @kaxpby!(n, -c, v, s, w̄)

    xlqNorm² += ζ * ζ
    xlqNorm = sqrt(xlqNorm²)

    # check stopping condition based on forward error lower bound
    err_vec[mod(iter, window) + 1] = ζ
    if iter ≥ window
      err_lbnd = norm(err_vec)
      history && push!(err_lbnds, err_lbnd)
      fwd_err_lbnd = err_lbnd ≤ etol * xlqNorm
    end

    # compute LQ forward error upper bound
    if σ > 0 && !complex_error_bnd
      η̃ = ω * s
      ϵ̃ = -ω * c
      τ̃ = -τ * δ / ω
      ζ̃ = (τ̃ - ζ * η̃) / ϵ̃
      history && push!(err_ubnds_lq, abs(ζ̃ ))
    end

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    ill_cond_mach = (one(T) + test3 ≤ one(T))
    solved_mach = (one(T) + test2 ≤ one(T))
    zero_resid_mach = (one(T) + t1 ≤ one(T))

    # Stopping conditions based on user-provided tolerances.
    tired  = iter ≥ itmax
    ill_cond_lim = (test3 ≤ ctol)
    solved_lim = (test2 ≤ atol)
    zero_resid_lim = (test1 ≤ rtol)

    ill_cond = ill_cond_mach || ill_cond_lim
    solved = solved_mach || solved_lim || zero_resid_mach || zero_resid_lim || fwd_err_lbnd || fwd_err_ubnd

    iter = iter + 1
  end
  (verbose > 0) && @printf("\n")

  if transfer_to_lsqr  # compute LSQR point
    @kaxpy!(n, ζ̄ , w̄, x)
  end

  tired         && (status = "maximum number of iterations exceeded")
  ill_cond_mach && (status = "condition number seems too large for this machine")
  ill_cond_lim  && (status = "condition number exceeds tolerance")
  solved        && (status = "found approximate minimum least-squares solution")
  zero_resid    && (status = "found approximate zero-residual solution")
  fwd_err_lbnd  && (status = "forward error lower bound small enough")
  fwd_err_ubnd  && (status = "forward error upper bound small enough")

  # Update stats
  stats.solved = solved
  stats.inconsistent = !zero_resid
  stats.error_with_bnd = complex_error_bnd
  stats.status = status
  return solver
end
