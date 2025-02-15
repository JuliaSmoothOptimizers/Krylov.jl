# An implementation of SYMMLQ for the solution of the
# linear system Ax = b, where A is Hermitian.
#
# This implementation follows the original implementation by
# Michael Saunders described in
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Ron Estrin, <ronestrin756@gmail.com>

export symmlq, symmlq!

"""
    (x, stats) = symmlq(A, b::AbstractVector{FC};
                        M=I, ldiv::Bool=false, window::Int=5,
                        transfer_to_cg::Bool=true, λ::T=zero(T),
                        λest::T=zero(T), etol::T=√eps(T),
                        conlim::T=1/√eps(T), atol::T=√eps(T),
                        rtol::T=√eps(T), itmax::Int=0,
                        timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                        callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = symmlq(A, b, x0::AbstractVector; kwargs...)

SYMMLQ can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above

Solve the shifted linear system

    (A + λI) x = b

of size n using the SYMMLQ method, where λ is a shift parameter, and A is Hermitian.

SYMMLQ produces monotonic errors ‖x* - x‖₂.

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `window`: number of iterations used to accumulate a lower bound on the error;
* `transfer_to_cg`: transfer from the SYMMLQ point to the CG point, when it exists. The transfer is based on the residual norm;
* `λ`: regularization parameter;
* `λest`: positive strict lower bound on the smallest eigenvalue `λₘᵢₙ` when solving a positive-definite system, such as `λest = (1-10⁻⁷)λₘᵢₙ`;
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

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SymmlqStats`](@ref) structure.

#### Reference

* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function symmlq end

"""
    solver = symmlq!(solver::SymmlqSolver, A, b; kwargs...)
    solver = symmlq!(solver::SymmlqSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`symmlq`](@ref).

See [`SymmlqSolver`](@ref) for more details about the `solver`.
"""
function symmlq! end

def_args_symmlq = (:(A                    ),
                   :(b::AbstractVector{FC}))

def_optargs_symmlq = (:(x0::AbstractVector),)

def_kwargs_symmlq = (:(; M = I                      ),
                     :(; ldiv::Bool = false         ),
                     :(; transfer_to_cg::Bool = true),
                     :(; λ::T = zero(T)             ),
                     :(; λest::T = zero(T)          ),
                     :(; atol::T = √eps(T)          ),
                     :(; rtol::T = √eps(T)          ),
                     :(; etol::T = √eps(T)          ),
                     :(; conlim::T = 1/√eps(T)      ),
                     :(; itmax::Int = 0             ),
                     :(; timemax::Float64 = Inf     ),
                     :(; verbose::Int = 0           ),
                     :(; history::Bool = false      ),
                     :(; callback = solver -> false ),
                     :(; iostream::IO = kstdout     ))

def_kwargs_symmlq = extract_parameters.(def_kwargs_symmlq)

args_symmlq = (:A, :b)
optargs_symmlq = (:x0,)
kwargs_symmlq = (:M, :ldiv, :transfer_to_cg, :λ, :λest, :atol, :rtol, :etol, :conlim, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function symmlq!(solver :: SymmlqSolver{T,FC,S}, $(def_args_symmlq...); $(def_kwargs_symmlq...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "SYMMLQ: system of size %d\n", n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :v, S, solver.x)  # The length of v is n
    x, Mvold, Mv, Mv_next, w̅ = solver.x, solver.Mvold, solver.Mv, solver.Mv_next, solver.w̅
    Δx, clist, zlist, sprod, stats = solver.Δx, solver.clist, solver.zlist, solver.sprod, solver.stats
    warm_start = solver.warm_start
    rNorms, rcgNorms = stats.residuals, stats.residualscg
    errors, errorscg = stats.errors, stats.errorscg
    reset!(stats)
    v = MisI ? Mv : solver.v
    vold = MisI ? Mvold : solver.v

    ϵM = eps(T)
    ctol = conlim > 0 ? 1 / conlim : zero(T)

    # Initial solution x₀
    kfill!(x, zero(FC))

    if warm_start
      mul!(Mvold, A, Δx)
      (λ ≠ 0) && kaxpy!(n, λ, Δx, Mvold)
      kaxpby!(n, one(FC), b, -one(FC), Mvold)
    else
      kcopy!(n, Mvold, b)  # Mvold ← b
    end

    # Initialize Lanczos process.
    # β₁ M v₁ = b.
    MisI || mulorldiv!(vold, M, Mvold, ldiv)
    β₁ = kdotr(m, vold, Mvold)
    if β₁ == 0
      stats.niter = 0
      stats.solved = true
      stats.Anorm = T(NaN)
      stats.Acond = T(NaN)
      history && push!(rNorms, zero(T))
      history && push!(rcgNorms, zero(T))
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      solver.warm_start = false
      return solver
    end
    β₁ = sqrt(β₁)
    β = β₁
    kscal!(m, one(FC) / β, vold)
    MisI || kscal!(m, one(FC) / β, Mvold)

    kcopy!(n, w̅, vold)  # w̅ ← vold

    mul!(Mv, A, vold)
    α = kdotr(m, vold, Mv) + λ
    kaxpy!(m, -α, Mvold, Mv)  # Mv = Mv - α * Mvold
    MisI || mulorldiv!(v, M, Mv, ldiv)
    β = kdotr(m, v, Mv)
    β < 0 && error("Preconditioner is not positive definite")
    β = sqrt(β)
    kscal!(m, one(FC) / β, v)
    MisI || kscal!(m, one(FC) / β, Mv)

    # Start QR factorization
    γbar = α
    δbar = β
    ϵold = zero(T)
    cold = one(T)
    sold = zero(T)

    ηold = zero(T)
    η    = β₁
    ζold = zero(T)

    ANorm² = α * α + β * β

    γmax = T(-Inf)
    γmin = T(Inf)
    ANorm = zero(T)
    Acond = zero(T)

    xNorm = zero(T)
    rNorm = β₁
    history && push!(rNorms, rNorm)

    if γbar ≠ 0
      ζbar = η / γbar
      xcgNorm = abs(ζbar)
      rcgNorm = β₁ * abs(ζbar)
      history && push!(rcgNorms, rcgNorm)
    else
      history && push!(rcgNorms, missing)
    end

    err = T(Inf)
    errcg = T(Inf)

    window = length(clist)
    kfill!(clist, zero(T))
    kfill!(zlist, zero(T))
    kfill!(sprod, one(T))

    if λest ≠ 0
      # Start QR factorization of Tₖ - λest I
      ρbar = α - λest
      σbar = β
      ρ = sqrt(ρbar * ρbar + β * β)
      cwold = -one(T)
      cw = ρbar / ρ
      sw = β / ρ

      history && push!(errors, abs(β₁ / λest))
      if γbar ≠ 0
        history && push!(errorscg, sqrt(errors[1]^2 - ζbar^2))
      else
        history && push!(errorscg, missing)
      end
    end

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %8s  %8s  %7s  %7s  %7s  %5s\n", "k", "‖r‖", "β", "cos", "sin", "‖A‖", "κ(A)", "test1", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7s  %.2fs\n", iter, rNorm, β, cold, sold, ANorm, Acond, "✗ ✗ ✗ ✗", start_time |> ktimer)

    tol = atol + rtol * β₁
    status = "unknown"
    solved_lq = solved_mach = solved_lim = (rNorm ≤ tol)
    solved_cg = (γbar ≠ 0) && transfer_to_cg && rcgNorm ≤ tol
    tired = iter ≥ itmax
    ill_cond = ill_cond_mach = ill_cond_lim = false
    solved = zero_resid = solved_lq || solved_cg
    fwd_err = false
    user_requested_exit = false
    overtimed = false

    while ! (solved || tired || ill_cond || user_requested_exit || overtimed)
      iter = iter + 1

      # Continue QR factorization
      (c, s, γ) = sym_givens(γbar, β)

      # Update SYMMLQ point
      ηold = η
      ζ = ηold / γ
      kaxpy!(n, c * ζ, w̅, x)
      kaxpy!(n, s * ζ, v, x)
      # Update w̅
      kaxpby!(n, -c, v, s, w̅)

      # Generate next Lanczos vector
      oldβ = β
      mul!(Mv_next, A, v)
      α = kdotr(m, v, Mv_next) + λ
      kaxpy!(m, -oldβ, Mvold, Mv_next)
      kcopy!(m, Mvold, Mv)  # Mvold ← Mv
      kaxpy!(m, -α, Mv, Mv_next)
      kcopy!(m, Mv, Mv_next)  # Mv ← Mv_next
      MisI || mulorldiv!(v, M, Mv, ldiv)
      β = kdotr(m, v, Mv)
      β < 0 && error("Preconditioner is not positive definite")
      β = sqrt(β)
      kscal!(m, one(FC) / β, v)
      MisI || kscal!(m, one(FC) / β, Mv)

      # Continue A norm estimate
      ANorm² = ANorm² + α * α + oldβ * oldβ + β * β

      if λest ≠ 0
        η = -oldβ * oldβ * cwold / ρbar
        ω = λest + η
        ψ = c * δbar + s * ω
        ωbar = s * δbar - c * ω
      end

      # Continue QR factorization
      δ = δbar * c + α * s
      γbar = δbar * s - α * c
      ϵ = β * s
      δbar = -β * c
      η = -ϵold * ζold - δ * ζ

      rNorm = sqrt(γ * γ * ζ * ζ + ϵold * ϵold * ζold * ζold)
      xNorm = xNorm + ζ * ζ
      history && push!(rNorms, rNorm)

      if γbar ≠ 0
        ζbar = η / γbar
        rcgNorm = β * abs(s * ζ - c * ζbar)
        xcgNorm = xNorm + ζbar * ζbar
        history && push!(rcgNorms, rcgNorm)
      else
        history && push!(rcgNorms, missing)
      end

      if window > 0 && λest ≠ 0
        if iter < window && window > 1
          for i = iter+1 : window
            sprod[i] = s * sprod[i]
          end
        end

        ix = ((iter-1) % window) + 1
        clist[ix] = c
        zlist[ix] = ζ

        if iter ≥ window
            jx = mod(iter, window) + 1
            zetabark = zlist[jx] / clist[jx]

            if γbar ≠ 0
              theta = zero(T)
              for i = 1 : window
                theta += clist[i] * sprod[i] * zlist[i]
              end
              theta = zetabark * abs(theta) + abs(zetabark * ζbar * sprod[ix] * s) - zetabark^2
              history && (errorscg[iter-window+1] = sqrt(abs(errorscg[iter-window+1]^2 - 2*theta)))
            else
              history && (errorscg[iter-window+1] = missing)
            end
        end

        ix = (iter % window) + 1
        if iter ≥ window && window > 1
           sprod .= sprod ./ sprod[(ix % window) + 1]
           sprod[ix] = sprod[mod(ix-2, window)+1] * s
        end
      end

      if λest ≠ 0
        err = abs((ϵold * ζold + ψ * ζ) / ωbar)
        history && push!(errors, err)

        if γbar ≠ 0
          errcg = sqrt(abs(err * err - ζbar * ζbar))
          history && push!(errorscg, errcg)
        else
          history && push!(errorscg, missing)
        end

        ρbar = sw * σbar - cw * (α - λest)
        σbar = -cw * β
        ρ = sqrt(ρbar * ρbar + β * β)

        cwold = cw

        cw = ρbar / ρ
        sw = β / ρ
      end

      # TODO: Use γ or γbar?
      γmax = max(γmax, γ)
      γmin = min(γmin, γ)

      Acond = γmax / γmin
      ANorm = sqrt(ANorm²)
      test1 = rNorm / (ANorm * xNorm)

      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, β, c, s, ANorm, Acond, test1, start_time |> ktimer)

      # Reset variables
      ϵold = ϵ
      ζold = ζ
      cold = c

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (one(T) + rNorm ≤ one(T))
      ill_cond_mach = (one(T) + one(T) / Acond ≤ one(T))
      zero_resid_mach = (one(T) + test1 ≤ one(T))
      # solved_mach = (ϵx ≥ β₁)

      # Stopping conditions based on user-provided tolerances.
      tired = iter ≥ itmax
      ill_cond_lim = (one(T) / Acond ≤ ctol)
      zero_resid_lim = (test1 ≤ tol)
      fwd_err = (err ≤ etol) || ((γbar ≠ 0) && (errcg ≤ etol))
      solved_lq = rNorm ≤ tol
      solved_cg = transfer_to_cg && (γbar ≠ 0) && rcgNorm ≤ tol
      
      user_requested_exit = callback(solver) :: Bool
      zero_resid = solved_lq || solved_cg
      ill_cond = ill_cond_mach || ill_cond_lim
      solved = solved_mach || zero_resid || zero_resid_mach || zero_resid_lim || fwd_err || resid_decrease_mach
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute CG point
    # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * w̅ₖ
    if solved_cg
      kaxpy!(m, ζbar, w̅, x)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    solved              && (status = "found approximate solution")
    solved_lq           && (status = "solution xᴸ good enough given atol and rtol")
    solved_cg           && (status = "solution xᶜ good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.Anorm = ANorm
    stats.Acond = Acond
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
