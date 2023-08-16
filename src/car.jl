# An implementation of CAR for the solution of hermitian positive definite linear systems.
#
# This method is described in
#
# A. Montoison, D. Orban and M. A. Saunders
# MinAres: An Iterative Solver for Symmetric Linear Systems
# Cahier du GERAD G-2023-XX.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Osaka, August 2023.

export car, car!

"""
    (x, stats) = car(A, b::AbstractVector{FC};
                     M=I, ldiv::Bool=false,
                     atol::T=√eps(T), rtol::T=√eps(T),
                     itmax::Int=0, timemax::Float64=Inf,
                     verbose::Int=0, history::Bool=false,
                     callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = car(A, b, x0::AbstractVector; kwargs...)

CAR can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

CAR solves the Hermitian linear system Ax = b of size n.

#### Input arguments

* `A`: a linear operator that models a Hermitian positive definite matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
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
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference
* A. Montoison, D. Orban and M. A. Saunders, [*MinAres: An Iterative Solver for Symmetric Linear Systems*](https://dx.doi.org/...), Cahier du GERAD G-2023-XX, GERAD, Montréal, 2023.
"""
function car end

"""
    solver = car!(solver::CarSolver, A, b; kwargs...)
    solver = car!(solver::CarSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`car`](@ref).

See [`CarSolver`](@ref) for more details about the `solver`.
"""
function car! end

def_args_car = (:(A                    ),
                :(b::AbstractVector{FC}))

def_optargs_car = (:(x0::AbstractVector),)

def_kwargs_car = (:(; M = I                     ),
                  :(; ldiv::Bool = false        ),
                  :(; atol::T = √eps(T)         ),
                  :(; rtol::T = √eps(T)         ),
                  :(; itmax::Int = 0            ),
                  :(; timemax::Float64 = Inf    ),
                  :(; verbose::Int = 0          ),
                  :(; history::Bool = false     ),
                  :(; callback = solver -> false),
                  :(; iostream::IO = kstdout    ))

def_kwargs_car = mapreduce(extract_parameters, vcat, def_kwargs_car)

args_car = (:A, :b)
optargs_car = (:x0,)
kwargs_car = (:M, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function car($(def_args_car...), $(def_optargs_car...); $(def_kwargs_car...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CarSolver(A, b)
    warm_start!(solver, $(optargs_car...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    car!(solver, $(args_car...); $(kwargs_car...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function car($(def_args_car...); $(def_kwargs_car...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = CarSolver(A, b)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    car!(solver, $(args_car...); $(kwargs_car...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.stats)
  end

  function car!(solver :: CarSolver{T,FC,S}, $(def_args_car...); $(def_kwargs_car...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CAR: system of %d equations in %d variables\n", n, n)

    # Tests M = Iₙ
    MisI = (M === I)
    !MisI && error("The support of a preconditioner is not implemented in CAR.")

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")

    # Set up workspace.
    Δx, x, r, p, s, q, t, u, stats = solver.Δx, solver.x, solver.r, solver.p, solver.s, solver.q, solver.t, solver.u, solver.stats
    warm_start = solver.warm_start
    rNorms, ArNorms = stats.residuals, stats.Aresiduals
    reset!(stats)

    x .= zero(FC)
    if warm_start
      mul!(r, A, Δx)
      @kaxpby!(n, one(FC), b, -one(FC), r)
    else
      r .= b
    end
    p .= r               # p₀ = r₀
    mul!(s, A, r)        # s₀ = Ar₀
    q .= s               # q₀ = Ap₀
    mul!(t, A, s)        # t₀ = A²r₀
    u .= t               # u₀ = A²p₀
    ρ = @kdotr(n, t, s)  # ρ₀ = ⟨t₀ , s₀⟩

    # Compute ‖r₀‖
    rNorm = @knrm2(n, r)
    history && push!(rNorms, rNorm)

    # Compute ‖Ar₀‖
    ArNorm = @knrm2(n, s)
    history && push!(ArNorms, ArNorm)

    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %7s  %5s\n", "k", "‖rₖ‖", "‖Arₖ‖", "α", "β", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7s  %7s  %.2fs\n", iter, rNorm, ArNorm, "✗ ✗ ✗ ✗", "✗ ✗ ✗ ✗", ktimer(start_time))

    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    inconsistent = false
    user_requested_exit = false
    overtimed = false
    status = "unknown"

    while !(solved || tired || user_requested_exit || overtimed)

      α = ρ / @kdotr(n, u, u)  # αₖ = ρₖ / ‖uₖ‖²
      @kaxpy!(n,  α, p, x)     # xₖ₊₁ = xₖ + αₖ * pₖ
      @kaxpy!(n, -α, q, r)     # rₖ₊₁ = rₖ - αₖ * qₖ
      @kaxpy!(n, -α, u, s)     # sₖ₊₁ = sₖ - αₖ * uₖ

      # Compute ‖rₖ‖
      rNorm = @knrm2(n, r)
      history && push!(rNorms, rNorm)

      # Compute ‖Arₖ‖
      ArNorm = @knrm2(n, s)
      history && push!(ArNorms, ArNorm)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach

      if !solved
        mul!(t, A, s)                  # tₖ₊₁ = A * sₖ₊₁
        ρ_next = @kdotr(n, t, s)       # ρₖ₊₁ = ⟨tₖ₊₁ , sₖ₊₁⟩
        β = ρ_next / ρ                 # βₖ = ρₖ₊₁ / ρₖ
        ρ = ρ_next
        @kaxpby!(n, one(FC), r, β, p)  # pₖ₊₁ = rₖ₊₁ + βₖ * pₖ 
        @kaxpby!(n, one(FC), s, β, q)  # qₖ₊₁ = sₖ₊₁ + βₖ * qₖ
        @kaxpby!(n, one(FC), t, β, u)  # uₖ₊₁ = tₖ₊₁ + βₖ * uₖ
      end

      iter = iter + 1
      tired = iter ≥ itmax
      user_requested_exit = callback(solver) :: Bool
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && !solved && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, ArNorm, α, β, ktimer(start_time))
      kdisplay(iter, verbose) &&  solved && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %7s  %.2fs\n", iter, rNorm, ArNorm, α, "✗ ✗ ✗ ✗", ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    solved              && (status = "solution good enough given atol and rtol")
    tired               && (status = "maximum number of iterations exceeded")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && @kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = inconsistent
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
