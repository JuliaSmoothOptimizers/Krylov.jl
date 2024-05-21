# An implementation of the Lanczos version of the conjugate gradient method.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
# SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cg_lanczos, cg_lanczos!

"""
    (x, stats) = cg_lanczos(A, b::AbstractVector{FC};
                            M=I, ldiv::Bool=false,
                            check_curvature::Bool=false, atol::T=√eps(T),
                            rtol::T=√eps(T), itmax::Int=0,
                            timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                            callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = cg_lanczos(A, b, x0::AbstractVector; kwargs...)

CG-LANCZOS can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

The Lanczos version of the conjugate gradient method to solve the
Hermitian linear system Ax = b of size n.

The method does _not_ abort if A is not definite.

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `check_curvature`: if `true`, check that the curvature of the quadratic along the search direction is positive, and abort if not, unless `linesearch` is also `true`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`LanczosStats`](@ref) structure.

#### References

* A. Frommer and P. Maass, [*Fast CG-Based Methods for Tikhonov-Phillips Regularization*](https://doi.org/10.1137/S1064827596313310), SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function cg_lanczos end

"""
    solver = cg_lanczos!(solver::CgLanczosSolver, A, b; kwargs...)
    solver = cg_lanczos!(solver::CgLanczosSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`cg_lanczos`](@ref).

See [`CgLanczosSolver`](@ref) for more details about the `solver`.
"""
function cg_lanczos! end

def_args_cg_lanczos = (:(A                    ),
                       :(b::AbstractVector{FC}))

def_optargs_cg_lanczos = (:(x0::AbstractVector),)

def_kwargs_cg_lanczos = (:(; M = I                        ),
                         :(; ldiv::Bool = false           ),
                         :(; check_curvature::Bool = false),
                         :(; atol::T = √eps(T)            ),
                         :(; rtol::T = √eps(T)            ),
                         :(; itmax::Int = 0               ),
                         :(; timemax::Float64 = Inf       ),
                         :(; verbose::Int = 0             ),
                         :(; history::Bool = false        ),
                         :(; callback = solver -> false   ),
                         :(; iostream::IO = kstdout       ))

def_kwargs_cg_lanczos = mapreduce(extract_parameters, vcat, def_kwargs_cg_lanczos)

args_cg_lanczos = (:A, :b)
optargs_cg_lanczos = (:x0,)
kwargs_cg_lanczos = (:M, :ldiv, :check_curvature, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cg_lanczos($(def_args_cg_lanczos...), $(def_optargs_cg_lanczos...); $(def_kwargs_cg_lanczos...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CgLanczosSolver(A, b)
    warm_start!(solver, $(optargs_cg_lanczos...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    cg_lanczos!(solver, $(args_cg_lanczos...); $(kwargs_cg_lanczos...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function cg_lanczos($(def_args_cg_lanczos...); $(def_kwargs_cg_lanczos...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CgLanczosSolver(A, b)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    cg_lanczos!(solver, $(args_cg_lanczos...); $(kwargs_cg_lanczos...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function cg_lanczos!(solver :: CgLanczosSolver{T,FC,S}, $(def_args_cg_lanczos...); $(def_kwargs_cg_lanczos...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CG-LANCZOS: system of %d equations in %d variables\n", n, n)

    # Tests M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :v, S, n)
    Δx, x, Mv, Mv_prev = solver.Δx, solver.x, solver.Mv, solver.Mv_prev
    p, Mv_next, stats = solver.p, solver.Mv_next, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    v = MisI ? Mv : solver.v

    # Initial state.
    x .= zero(FC)
    if warm_start
      mul!(Mv, A, Δx)
      @kaxpby!(n, one(FC), b, -one(FC), Mv)
    else
      Mv .= b
    end
    MisI || mulorldiv!(v, M, Mv, ldiv)  # v₁ = M⁻¹r₀
    β = sqrt(@kdotr(n, v, Mv))          # β₁ = v₁ᴴ M v₁
    σ = β
    rNorm = σ
    history && push!(rNorms, rNorm)
    if β == 0
      stats.niter = 0
      stats.solved = true
      stats.Anorm = zero(T)
      stats.indefinite = false
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end
    p .= v

    # Initialize Lanczos process.
    # β₁Mv₁ = b
    @kscal!(n, one(FC) / β, v)           # v₁  ←  v₁ / β₁
    MisI || @kscal!(n, one(FC) / β, Mv)  # Mv₁ ← Mv₁ / β₁
    Mv_prev .= Mv

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    # Initialize some constants used in recursions below.
    ω = zero(T)
    γ = one(T)
    Anorm2 = zero(T)
    β_prev = zero(T)

    # Define stopping tolerance.
    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))

    indefinite = false
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    # Main loop.
    while ! (solved || tired || (check_curvature & indefinite) || user_requested_exit || overtimed)
      # Form next Lanczos vector.
      # βₖ₊₁Mvₖ₊₁ = Avₖ - δₖMvₖ - βₖMvₖ₋₁
      mul!(Mv_next, A, v)        # Mvₖ₊₁ ← Avₖ
      δ = @kdotr(n, v, Mv_next)  # δₖ = vₖᴴ A vₖ

      # Check curvature. Exit fast if requested.
      # It is possible to show that σₖ² (δₖ - ωₖ₋₁ / γₖ₋₁) = pₖᴴ A pₖ.
      γ = one(T) / (δ - ω / γ)  # γₖ = 1 / (δₖ - ωₖ₋₁ / γₖ₋₁)
      indefinite |= (γ ≤ 0)
      (check_curvature & indefinite) && continue

      @kaxpy!(n, -δ, Mv, Mv_next)        # Mvₖ₊₁ ← Mvₖ₊₁ - δₖMvₖ
      if iter > 0
        @kaxpy!(n, -β, Mv_prev, Mv_next) # Mvₖ₊₁ ← Mvₖ₊₁ - βₖMvₖ₋₁
        @kcopy!(n, Mv, Mv_prev)          # Mvₖ₋₁ ← Mvₖ
      end
      @kcopy!(n, Mv_next, Mv)              # Mvₖ ← Mvₖ₊₁
      MisI || mulorldiv!(v, M, Mv, ldiv)   # vₖ₊₁ = M⁻¹ * Mvₖ₊₁
      β = sqrt(@kdotr(n, v, Mv))           # βₖ₊₁ = vₖ₊₁ᴴ M vₖ₊₁
      @kscal!(n, one(FC) / β, v)           # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
      MisI || @kscal!(n, one(FC) / β, Mv)  # Mvₖ₊₁ ← Mvₖ₊₁ / βₖ₊₁
      Anorm2 += β_prev^2 + β^2 + δ^2       # Use ‖Tₖ₊₁‖₂ as increasing approximation of ‖A‖₂.
      β_prev = β

      # Compute next CG iterate.
      @kaxpy!(n, γ, p, x)     # xₖ₊₁ = xₖ + γₖ * pₖ
      ω = β * γ
      σ = -ω * σ              # σₖ₊₁ = - βₖ₊₁ * γₖ * σₖ
      ω = ω * ω               # ωₖ = (βₖ₊₁ * γₖ)²
      @kaxpby!(n, σ, v, ω, p) # pₖ₊₁ = σₖ₊₁ * vₖ₊₁ + ωₖ * pₖ
      rNorm = abs(σ)          # ‖rₖ₊₁‖_M = |σₖ₊₁| because rₖ₊₁ = σₖ₊₁ * vₖ₊₁ and ‖vₖ₊₁‖_M = 1
      history && push!(rNorms, rNorm)
      iter = iter + 1
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired                          && (status = "maximum number of iterations exceeded")
    (check_curvature & indefinite) && (status = "negative curvature")
    solved                         && (status = "solution good enough given atol and rtol")
    user_requested_exit            && (status = "user-requested exit")
    overtimed                      && (status = "time limit exceeded")

    # Update x
    warm_start && @kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats. TODO: Estimate Acond.
    stats.niter = iter
    stats.solved = solved
    stats.Anorm = sqrt(Anorm2)
    stats.indefinite = indefinite
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
