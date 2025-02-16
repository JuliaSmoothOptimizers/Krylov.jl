# An implementation of LSLQ for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# equivalently, of the normal equations
#
#  AᴴAx = Aᴴb.
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
                      M=I, N=I, ldiv::Bool=false,
                      window::Int=5, transfer_to_lsqr::Bool=false,
                      sqd::Bool=false, λ::T=zero(T),
                      σ::T=zero(T), etol::T=√eps(T),
                      utol::T=√eps(T), btol::T=√eps(T),
                      conlim::T=1/√eps(T), atol::T=√eps(T),
                      rtol::T=√eps(T), itmax::Int=0,
                      timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                      callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the regularized linear least-squares problem

    minimize ‖b - Ax‖₂² + λ²‖x‖₂²

of size m × n using the LSLQ method, where λ ≥ 0 is a regularization parameter.
LSLQ is formally equivalent to applying SYMMLQ to the normal equations

    (AᴴA + λ²I) x = Aᴴb

but is more stable.

If `λ > 0`, we solve the symmetric and quasi-definite system

    [ E      A ] [ r ]   [ b ]
    [ Aᴴ  -λ²F ] [ x ] = [ 0 ],

where E and F are symmetric and positive definite.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.
If `sqd=true`, `λ` is set to the common value `1`.

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹ + λ²‖x‖²_F.

For a symmetric and positive definite matrix `K`, the K-norm of a vector `x` is `‖x‖²_K = xᴴKx`.
LSLQ is then equivalent to applying SYMMLQ to `(AᴴE⁻¹A + λ²F)x = AᴴE⁻¹b` with `r = E⁻¹(b - Ax)`.

If `λ = 0`, we solve the symmetric and indefinite system

    [ E    A ] [ r ]   [ b ]
    [ Aᴴ   0 ] [ x ] = [ 0 ].

The system above represents the optimality conditions of

    minimize ‖b - Ax‖²_E⁻¹.

In this case, `N` can still be specified and indicates the weighted norm in which `x` and `Aᴴr` should be measured.
`r` can be recovered by computing `E⁻¹(b - Ax)`.

#### Main features

* the solution estimate is updated along orthogonal directions
* the norm of the solution estimate ‖xᴸₖ‖₂ is increasing
* the error ‖eₖ‖₂ := ‖xᴸₖ - x*‖₂ is decreasing
* it is possible to transition cheaply from the LSLQ iterate to the LSQR iterate if there is an advantage (there always is in terms of error)
* if `A` is rank deficient, identify the minimum least-squares solution

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the augmented system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the augmented system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `window`: number of iterations used to accumulate a lower bound on the error;
* `transfer_to_lsqr`: transfer from the LSLQ point to the LSQR point, when it exists. The transfer is based on the residual norm;
* `sqd`: if `true`, set `λ=1` for Hermitian quasi-definite systems;
* `λ`: regularization parameter;
* `σ`: strict lower bound on the smallest positive singular value `σₘᵢₙ` such as `σ = (1-10⁻⁷)σₘᵢₙ`;
* `etol`: stopping tolerance based on the lower bound on the error;
* `utol`: stopping tolerance based on the upper bound on the error;
* `btol`: stopping tolerance used to detect zero-residual problems;
* `conlim`: limit on the estimated condition number of `A` beyond which the solution will be abandoned;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`LSLQStats`](@ref) structure.

* `stats.err_lbnds` is a vector of lower bounds on the LQ error---the vector is empty if `window` is set to zero
* `stats.err_ubnds_lq` is a vector of upper bounds on the LQ error---the vector is empty if `σ == 0` is left at zero
* `stats.err_ubnds_cg` is a vector of upper bounds on the CG error---the vector is empty if `σ == 0` is left at zero
* `stats.error_with_bnd` is a boolean indicating whether there was an error in the upper bounds computation (cancellation errors, too large σ ...)

#### Stopping conditions

The iterations stop as soon as one of the following conditions holds true:

* the optimality residual is sufficiently small (`stats.status = "found approximate minimum least-squares solution"`) in the sense that either
  * ‖Aᴴr‖ / (‖A‖ ‖r‖) ≤ atol, or
  * 1 + ‖Aᴴr‖ / (‖A‖ ‖r‖) ≤ 1
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
function lslq end

"""
    solver = lslq!(solver::LslqSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`lslq`](@ref).

See [`LslqSolver`](@ref) for more details about the `solver`.
"""
function lslq! end

def_args_lslq = (:(A                    ),
                 :(b::AbstractVector{FC}))

def_kwargs_lslq = (:(; M = I                         ),
                   :(; N = I                         ),
                   :(; ldiv::Bool = false            ),
                   :(; transfer_to_lsqr::Bool = false),
                   :(; sqd::Bool = false             ),
                   :(; λ::T = zero(T)                ),
                   :(; σ::T = zero(T)                ),
                   :(; etol::T = √eps(T)             ),
                   :(; utol::T = √eps(T)             ),
                   :(; btol::T = √eps(T)             ),
                   :(; conlim::T = 1/√eps(T)         ),
                   :(; atol::T = √eps(T)             ),
                   :(; rtol::T = √eps(T)             ),
                   :(; itmax::Int = 0                ),
                   :(; timemax::Float64 = Inf        ),
                   :(; verbose::Int = 0              ),
                   :(; history::Bool = false         ),
                   :(; callback = solver -> false    ),
                   :(; iostream::IO = kstdout        ))

def_kwargs_lslq = extract_parameters.(def_kwargs_lslq)

args_lslq = (:A, :b)
kwargs_lslq = (:M, :N, :ldiv, :transfer_to_lsqr, :sqd, :λ, :σ, :etol, :utol, :btol, :conlim, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function lslq!(solver :: LslqSolver{T,FC,S}, $(def_args_lslq...); $(def_kwargs_lslq...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "LSLQ: system of %d equations in %d variables\n", m, n)

    # Check sqd and λ parameters
    sqd && (λ ≠ 0) && error("sqd cannot be set to true if λ ≠ 0 !")
    sqd && (λ = one(T))

    # Tests M = Iₙ and N = Iₘ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :u, S, solver.Av)  # The length of u is m
    allocate_if(!NisI, solver, :v, S, solver.x)   # The length of v is n
    x, Nv, Aᴴu, w̄ = solver.x, solver.Nv, solver.Aᴴu, solver.w̄
    Mu, Av, err_vec, stats = solver.Mu, solver.Av, solver.err_vec, solver.stats
    rNorms, ArNorms, err_lbnds = stats.residuals, stats.Aresiduals, stats.err_lbnds
    err_ubnds_lq, err_ubnds_cg = stats.err_ubnds_lq, stats.err_ubnds_cg
    reset!(stats)
    u = MisI ? Mu : solver.u
    v = NisI ? Nv : solver.v

    λ² = λ * λ
    ctol = conlim > 0 ? 1/conlim : zero(T)

    kfill!(x, zero(FC))  # LSLQ point

    # Initialize Golub-Kahan process.
    # β₁ M u₁ = b.
    kcopy!(m, Mu, b)  # Mu ← b
    MisI || mulorldiv!(u, M, Mu, ldiv)
    β₁ = knorm_elliptic(m, u, Mu)
    if β₁ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.error_with_bnd = false
      history && push!(rNorms, zero(T))
      history && push!(ArNorms, zero(T))
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      return solver
    end
    β = β₁

    kscal!(m, one(FC)/β₁, u)
    MisI || kscal!(m, one(FC)/β₁, Mu)
    mul!(Aᴴu, Aᴴ, u)
    kcopy!(n, Nv, Aᴴu)  # Nv ← Aᴴu
    NisI || mulorldiv!(v, N, Nv, ldiv)
    α = knorm_elliptic(n, v, Nv)  # = α₁

    # Aᴴb = 0 so x = 0 is a minimum least-squares solution
    if α == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.error_with_bnd = false
      history && push!(rNorms, β₁)
      history && push!(ArNorms, zero(T))
      stats.timer = start_time |> ktimer
      stats.status = "x is a minimum least-squares solution"
      return solver
    end
    kscal!(n, one(FC)/α, v)
    NisI || kscal!(n, one(FC)/α, Nv)

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

    kcopy!(n, w̄, v)  # w̄₁ = v₁

    err_lbnd = zero(T)
    window = length(err_vec)
    kfill!(err_vec, zero(T))
    complex_error_bnd = false

    # Initialize other constants.
    αL = α
    βL = β
    ρ̄ = -σ
    γ̄ = α
    ψ = β₁
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

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s  %5s\n", "k", "‖r‖", "‖Aᴴr‖", "β", "α", "cos", "sin", "‖A‖²", "κ(A)", "‖xL‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, ArNorm, β, α, c, s, Anorm², Acond, xlqNorm, start_time |> ktimer)

    status = "unknown"
    ε = atol + rtol * β₁
    solved = solved_mach = solved_lim = (rNorm ≤ ε)
    tired  = iter ≥ itmax
    ill_cond = ill_cond_mach = ill_cond_lim = false
    zero_resid = zero_resid_mach = zero_resid_lim = false
    fwd_err_lbnd = false
    fwd_err_ubnd = false
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || ill_cond || user_requested_exit || overtimed)

      # Generate next Golub-Kahan vectors.
      # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
      mul!(Av, A, v)
      kaxpby!(m, one(FC), Av, -α, Mu)
      MisI || mulorldiv!(u, M, Mu, ldiv)
      β = knorm_elliptic(m, u, Mu)
      if β ≠ 0
        kscal!(m, one(FC)/β, u)
        MisI || kscal!(m, one(FC)/β, Mu)

        # 2. αₖ₊₁Nvₖ₊₁ = Aᴴuₖ₊₁ - βₖ₊₁Nvₖ
        mul!(Aᴴu, Aᴴ, u)
        kaxpby!(n, one(FC), Aᴴu, -β, Nv)
        NisI || mulorldiv!(v, N, Nv, ldiv)
        α = knorm_elliptic(n, v, Nv)
        if α ≠ 0
          kscal!(n, one(FC)/α, v)
          NisI || kscal!(n, one(FC)/α, Nv)
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
      # k+1 [ s'  -c' ] [ β   α⁺ ]   [     γ̄  ]
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
      rNorm = sqrt((ψ * cp - ζold * η)^2 + (ψ * sp)^2)
      history && push!(rNorms, rNorm)

      ArNorm = sqrt((γ * ϵ * ζ)^2 + (δ * η * ζold)^2)
      history && push!(ArNorms, ArNorm)

      # Compute ψₖ
      ψ = ψ * sp

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

      test1 = rNorm
      test2 = ArNorm / (Anorm * rNorm)
      test3 = 1 / Acond
      t1    = test1 / (one(T) + Anorm * xlqNorm)
      tol   = btol + atol * Anorm * xlqNorm / β₁

      # update LSLQ point for next iteration
      kaxpy!(n, c * ζ, w̄, x)
      kaxpy!(n, s * ζ, v, x)

      # compute w̄
      kaxpby!(n, -c, v, s, w̄)

      xlqNorm² += ζ * ζ
      xlqNorm = sqrt(xlqNorm²)

      # check stopping condition based on forward error lower bound
      err_vec[mod(iter, window) + 1] = ζ
      if iter ≥ window
        err_lbnd = knorm(window, err_vec)
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
      user_requested_exit = callback(solver) :: Bool
      tired  = iter ≥ itmax
      ill_cond_lim = (test3 ≤ ctol)
      solved_lim = (test2 ≤ atol)
      zero_resid_lim = (test1 ≤ ε)

      ill_cond = ill_cond_mach || ill_cond_lim
      zero_resid = zero_resid_mach || zero_resid_lim
      solved = solved_mach || solved_lim || zero_resid || fwd_err_lbnd || fwd_err_ubnd
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      iter = iter + 1
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, ArNorm, β, α, c, s, Anorm, Acond, xlqNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    if transfer_to_lsqr  # compute LSQR point
      kaxpy!(n, ζ̄ , w̄, x)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    solved              && (status = "found approximate minimum least-squares solution")
    zero_resid          && (status = "found approximate zero-residual solution")
    fwd_err_lbnd        && (status = "forward error lower bound small enough")
    fwd_err_ubnd        && (status = "forward error upper bound small enough")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = !zero_resid
    stats.error_with_bnd = complex_error_bnd
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
