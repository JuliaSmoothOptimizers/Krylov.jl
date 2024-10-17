# An implementation of MINRES for the solution of the
# linear system Ax = b, or the linear least-squares problem
#
#  minimize ‖Ax - b‖₂
#
# where A is Hermitian.
#
# MINRES is formally equivalent to applying the conjugate residuals method
# to Ax = b when A is positive definite, but is more general and also applies
# to the case where A is indefinite.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Brussels, Belgium, June 2015.
# Montréal, August 2015.

export minres, minres!

"""
    (x, stats) = minres(A, b::AbstractVector{FC};
                        M=I, ldiv::Bool=false, window::Int=5,
                        λ::T=zero(T), atol::T=√eps(T),
                        rtol::T=√eps(T), etol::T=√eps(T),
                        conlim::T=1/√eps(T), itmax::Int=0,
                        timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                        callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = minres(A, b, x0::AbstractVector; kwargs...)

MINRES can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the shifted linear least-squares problem

    minimize ‖b - (A + λI)x‖₂²

or the shifted linear system

    (A + λI) x = b

of size n using the MINRES method, where λ ≥ 0 is a shift parameter,
where A is Hermitian.

MINRES is formally equivalent to applying CR to Ax=b when A is positive
definite, but is typically more stable and also applies to the case where
A is indefinite.

MINRES produces monotonic residuals ‖r‖₂ and optimality residuals ‖Aᴴr‖₂.

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `window`: number of iterations used to accumulate a lower bound on the error;
* `λ`: regularization parameter;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `etol`: stopping tolerance based on the lower bound on the error;
* `conlim`: limit on the estimated condition number of `A` beyond which the solution will be abandoned;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function minres end

"""
    solver = minres!(solver::MinresSolver, A, b; kwargs...)
    solver = minres!(solver::MinresSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`minres`](@ref).

See [`MinresSolver`](@ref) for more details about the `solver`.
"""
function minres! end

def_args_minres = (:(A                    ),
                   :(b::AbstractVector{FC}))

def_optargs_minres = (:(x0::AbstractVector),)

def_kwargs_minres = (:(; M = I                     ),
                     :(; ldiv::Bool = false        ),
                     :(; λ::T = zero(T)            ),
                     :(; atol::T = √eps(T)         ),
                     :(; rtol::T = √eps(T)         ),
                     :(; etol::T = √eps(T)         ),
                     :(; conlim::T = 1/√eps(T)     ),
                     :(; itmax::Int = 0            ),
                     :(; timemax::Float64 = Inf    ),
                     :(; verbose::Int = 0          ),
                     :(; history::Bool = false     ),
                     :(; callback = solver -> false),
                     :(; iostream::IO = kstdout    ))

def_kwargs_minres = extract_parameters.(def_kwargs_minres)

args_minres = (:A, :b)
optargs_minres = (:x0,)
kwargs_minres = (:M, :ldiv, :λ, :atol, :rtol, :etol, :conlim, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function minres!(solver :: MinresSolver{T,FC,S}, $(def_args_minres...); $(def_kwargs_minres...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "MINRES: system of size %d\n", n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :v, S, n)
    Δx, x, r1, r2, w1, w2, y = solver.Δx, solver.x, solver.r1, solver.r2, solver.w1, solver.w2, solver.y
    err_vec, stats = solver.err_vec, solver.stats
    warm_start = solver.warm_start
    rNorms, ArNorms, Aconds = stats.residuals, stats.Aresiduals, stats.Acond
    reset!(stats)
    v = MisI ? r2 : solver.v

    ϵM = eps(T)
    ctol = conlim > 0 ? 1 / conlim : zero(T)

    # Initial solution x₀
    kfill!(x, zero(FC))

    if warm_start
      mul!(r1, A, Δx)
      (λ ≠ 0) && kaxpy!(n, λ, Δx, r1)
      kaxpby!(n, one(FC), b, -one(FC), r1)
    else
      kcopy!(n, r1, b)  # r1 ← b
    end

    # Initialize Lanczos process.
    # β₁ M v₁ = b.
    kcopy!(n, r2, r1)  # r2 ← r1
    MisI || mulorldiv!(v, M, r1, ldiv)
    β₁ = kdotr(m, r1, v)
    β₁ < 0 && error("Preconditioner is not positive definite")
    if β₁ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      history && push!(rNorms, β₁)
      history && push!(ArNorms, zero(T))
      history && push!(Aconds, zero(T))
      solver.warm_start = false
      return solver
    end
    β₁ = sqrt(β₁)
    β = β₁

    oldβ = zero(T)
    δbar = zero(T)
    ϵ = zero(T)
    rNorm = β₁
    history && push!(rNorms, β₁)
    ϕbar = β₁
    rhs1 = β₁
    rhs2 = zero(T)
    γmax = zero(T)
    γmin = T(Inf)
    cs = -one(T)
    sn = zero(T)
    kfill!(w1, zero(FC))
    kfill!(w2, zero(FC))

    ANorm² = zero(T)
    ANorm = zero(T)
    Acond = zero(T)
    history && push!(Aconds, Acond)
    ArNorm = zero(T)
    history && push!(ArNorms, ArNorm)
    xNorm = zero(T)

    xENorm² = zero(T)
    err_lbnd = zero(T)
    window = length(err_vec)
    kfill!(err_vec, zero(T))

    iter = 0
    itmax == 0 && (itmax = 2*n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %8s  %8s  %7s  %7s  %7s  %7s  %5s\n", "k", "‖r‖", "‖Aᴴr‖", "β", "cos", "sin", "‖A‖", "κ(A)", "test1", "test2", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7s  %7s  %.2fs\n", iter, rNorm, ArNorm, β, cs, sn, ANorm, Acond, "✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗", ktimer(start_time))

    ε = atol + rtol * β₁
    solved = solved_mach = solved_lim = (rNorm ≤ rtol)
    tired  = iter ≥ itmax
    ill_cond = ill_cond_mach = ill_cond_lim = false
    zero_resid = zero_resid_mach = zero_resid_lim = (rNorm ≤ ε)
    fwd_err = false
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || ill_cond || user_requested_exit || overtimed)
      iter = iter + 1

      # Generate next Lanczos vector.
      mul!(y, A, v)
      λ ≠ 0 && kaxpy!(n, λ, v, y)              # (y = y + λ * v)
      kscal!(n, one(FC) / β, y)
      iter ≥ 2 && kaxpy!(n, -β / oldβ, r1, y)  # (y = y - β / oldβ * r1)

      α = kdotr(n, v, y) / β
      kaxpy!(n, -α / β, r2, y)  # y = y - α / β * r2

      # Compute w.
      δ = cs * δbar + sn * α
      if iter == 1
        w = w2
      else
        iter ≥ 3 && kscal!(n, -ϵ, w1)
        w = w1
        kaxpy!(n, -δ, w2, w)
      end
      kaxpy!(n, one(FC) / β, v, w)

      kcopy!(n, r1, r2)  # r1 ← r2
      kcopy!(n, r2, y)   # r2 ← y
      MisI || mulorldiv!(v, M, r2, ldiv)
      oldβ = β
      β = kdotr(n, r2, v)
      β < 0 && error("Preconditioner is not positive definite")
      β = sqrt(β)
      ANorm² = ANorm² + α * α + oldβ * oldβ + β * β

      # Apply rotation to obtain
      #  [ δₖ    ϵₖ₊₁    ] = [ cs  sn ] [ δbarₖ  0    ]
      #  [ γbar  δbarₖ₊₁ ]   [ sn -cs ] [ αₖ     βₖ₊₁ ]
      γbar = sn * δbar - cs * α
      ϵ = sn * β
      δbar = -cs * β
      root = sqrt(γbar * γbar + δbar * δbar)
      ArNorm = ϕbar * root  # = ‖Aᴴrₖ₋₁‖
      history && push!(ArNorms, ArNorm)

      # Compute the next plane rotation.
      γ = sqrt(γbar * γbar + β * β)
      γ = max(γ, ϵM)
      cs = γbar / γ
      sn = β / γ
      ϕ = cs * ϕbar
      ϕbar = sn * ϕbar

      # Final update of w.
      kscal!(n, one(FC) / γ, w)

      # Update x.
      kaxpy!(n, ϕ, w, x)  # x = x + ϕ * w
      xENorm² = xENorm² + ϕ * ϕ

      # Update directions for x.
      if iter ≥ 2
        @kswap!(w1, w2)
      end

      # Compute lower bound on forward error.
      err_vec[mod(iter, window) + 1] = ϕ
      iter ≥ window && (err_lbnd = knorm(window, err_vec))

      γmax = max(γmax, γ)
      γmin = min(γmin, γ)
      ζ = rhs1 / γ
      rhs1 = rhs2 - δ * ζ
      rhs2 = -ϵ * ζ

      # Estimate various norms.
      ANorm = sqrt(ANorm²)
      xNorm = knorm(n, x)
      ϵA = ANorm * ϵM
      ϵx = ANorm * xNorm * ϵM
      ϵr = ANorm * xNorm * rtol
      d = γbar
      d == 0 && (d = ϵA)

      rNorm = ϕbar

      test1 = rNorm / (ANorm * xNorm)
      test2 = root / ANorm
      history && push!(rNorms, rNorm)

      Acond = γmax / γmin
      history && push!(Aconds, Acond)

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, ArNorm, β, cs, sn, ANorm, Acond, test1, test2, ktimer(start_time))

      if iter == 1 && β / β₁ ≤ 10 * ϵM
        # Aᴴb = 0 so x = 0 is a minimum least-squares solution
        stats.niter = 1
        stats.solved, stats.inconsistent = true, true
        stats.storage = sizeof(solver)
        stats.timer = ktimer(start_time)
        stats.status = "x is a minimum least-squares solution"
        solver.warm_start = false
        return solver
      end

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      ill_cond_mach = (one(T) + one(T) / Acond ≤ one(T))
      solved_mach = (one(T) + test2 ≤ one(T))
      zero_resid_mach = (one(T) + test1 ≤ one(T))
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      # solved_mach = (ϵx ≥ β₁)

      # Stopping conditions based on user-provided tolerances.
      tired = iter ≥ itmax
      ill_cond_lim = (one(T) / Acond ≤ ctol)
      solved_lim = (test2 ≤ ε)
      zero_resid_lim = MisI && (test1 ≤ eps(T))
      resid_decrease_lim = (rNorm ≤ ε)
      iter ≥ window && (fwd_err = err_lbnd ≤ etol * sqrt(xENorm²))

      user_requested_exit = callback(solver) :: Bool
      zero_resid = zero_resid_mach || zero_resid_lim
      resid_decrease = resid_decrease_mach || resid_decrease_lim
      ill_cond = ill_cond_mach || ill_cond_lim
      solved = solved_mach || solved_lim || zero_resid || fwd_err || resid_decrease
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    solved              && (status = "found approximate minimum least-squares solution")
    zero_resid          && (status = "found approximate zero-residual solution")
    fwd_err             && (status = "truncated forward error small enough")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = !zero_resid
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
