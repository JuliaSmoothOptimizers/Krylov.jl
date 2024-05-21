# An implementation of CGS for the solution of the square linear system Ax = b.
#
# This method is described in
#
# P. Sonneveld, CGS, A Fast Lanczos-Type Solver for Nonsymmetric Linear systems.
# SIAM Journal on Scientific and Statistical Computing, 10(1), pp. 36--52, 1989.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, October 2018.

export cgs, cgs!

"""
    (x, stats) = cgs(A, b::AbstractVector{FC};
                     c::AbstractVector{FC}=b, M=I, N=I,
                     ldiv::Bool=false, atol::T=√eps(T),
                     rtol::T=√eps(T), itmax::Int=0,
                     timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                     callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = cgs(A, b, x0::AbstractVector; kwargs...)

CGS can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

Solve the consistent linear system Ax = b of size n using CGS.
CGS requires two initial vectors `b` and `c`.
The relation `bᴴc ≠ 0` must be satisfied and by default `c = b`.

From "Iterative Methods for Sparse Linear Systems (Y. Saad)" :

«The method is based on a polynomial variant of the conjugate gradients algorithm.
Although related to the so-called bi-conjugate gradients (BCG) algorithm,
it does not involve adjoint matrix-vector multiplications, and the expected convergence
rate is about twice that of the BCG algorithm.

The Conjugate Gradient Squared algorithm works quite well in many cases.
However, one difficulty is that, since the polynomials are squared, rounding errors
tend to be more damaging than in the standard BCG algorithm. In particular, very
high variations of the residual vectors often cause the residual norms computed
to become inaccurate.

TFQMR and BICGSTAB were developed to remedy this difficulty.»

#### Input arguments

* `A`: a linear operator that models a matrix of dimension n;
* `b`: a vector of length n.

#### Optional argument

* `x0`: a vector of length n that represents an initial guess of the solution x.

#### Keyword arguments

* `c`: the second initial vector of length `n` required by the Lanczos biorthogonalization process;
* `M`: linear operator that models a nonsingular matrix of size `n` used for left preconditioning;
* `N`: linear operator that models a nonsingular matrix of size `n` used for right preconditioning;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
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

* P. Sonneveld, [*CGS, A Fast Lanczos-Type Solver for Nonsymmetric Linear systems*](https://doi.org/10.1137/0910004), SIAM Journal on Scientific and Statistical Computing, 10(1), pp. 36--52, 1989.
"""
function cgs end

"""
    solver = cgs!(solver::CgsSolver, A, b; kwargs...)
    solver = cgs!(solver::CgsSolver, A, b, x0; kwargs...)

where `kwargs` are keyword arguments of [`cgs`](@ref).

See [`CgsSolver`](@ref) for more details about the `solver`.
"""
function cgs! end

def_args_cgs = (:(A                    ),
                :(b::AbstractVector{FC}))

def_optargs_cgs = (:(x0::AbstractVector),)

def_kwargs_cgs = (:(; c::AbstractVector{FC} = b ),
                  :(; M = I                     ),
                  :(; N = I                     ),
                  :(; ldiv::Bool = false        ),
                  :(; atol::T = √eps(T)         ),
                  :(; rtol::T = √eps(T)         ),
                  :(; itmax::Int = 0            ),
                  :(; timemax::Float64 = Inf    ),
                  :(; verbose::Int = 0          ),
                  :(; history::Bool = false     ),
                  :(; callback = solver -> false),
                  :(; iostream::IO = kstdout    ))

def_kwargs_cgs = extract_parameters.(def_kwargs_cgs)

args_cgs = (:A, :b)
optargs_cgs = (:x0,)
kwargs_cgs = (:c, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function cgs!(solver :: CgsSolver{T,FC,S}, $(def_args_cgs...); $(def_kwargs_cgs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "CGS: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Set up workspace.
    allocate_if(!MisI, solver, :vw, S, n)
    allocate_if(!NisI, solver, :yz, S, n)
    Δx, x, r, u, p, q, ts, stats = solver.Δx, solver.x, solver.r, solver.u, solver.p, solver.q, solver.ts, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    t = s = solver.ts
    v = MisI ? t : solver.vw
    w = MisI ? s : solver.vw
    y = NisI ? p : solver.yz
    z = NisI ? u : solver.yz
    r₀ = MisI ? r : solver.ts

    if warm_start
      mul!(r₀, A, Δx)
      @kaxpby!(n, one(FC), b, -one(FC), r₀)
    else
      @kcopy!(n, r₀, b)  # r₀ ← b
    end

    @kfill!(x, zero(FC))                # x₀
    MisI || mulorldiv!(r, M, r₀, ldiv)  # r₀

    # Compute residual norm ‖r₀‖₂.
    rNorm = @knrm2(n, r)
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    # Compute ρ₀ = ⟨ r̅₀,r₀ ⟩
    ρ = @kdot(n, c, r)
    if ρ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = false, false
      stats.storage = sizeof(solver)
      stats.timer = ktimer(start_time)
      stats.status = "Breakdown bᴴc = 0"
      solver.warm_start =false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))

    @kcopy!(n, u, r)      # u₀
    @kcopy!(n, p, r)      # p₀
    @kfill!(q, zero(FC))  # q₋₁

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = false
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)

      NisI || mulorldiv!(y, N, p, ldiv)  # yₖ = N⁻¹pₖ
      mul!(t, A, y)                      # tₖ = Ayₖ
      MisI || mulorldiv!(v, M, t, ldiv)  # vₖ = M⁻¹tₖ
      σ = @kdot(n, c, v)                 # σₖ = ⟨ r̅₀,M⁻¹AN⁻¹pₖ ⟩
      α = ρ / σ                          # αₖ = ρₖ / σₖ
      @kcopy!(n, q, u)                   # qₖ = uₖ
      @kaxpy!(n, -α, v, q)               # qₖ = qₖ - αₖ * M⁻¹AN⁻¹pₖ
      @kaxpy!(n, one(FC), q, u)          # uₖ₊½ = uₖ + qₖ
      NisI || mulorldiv!(z, N, u, ldiv)  # zₖ = N⁻¹uₖ₊½
      @kaxpy!(n, α, z, x)                # xₖ₊₁ = xₖ + αₖ * N⁻¹(uₖ + qₖ)
      mul!(s, A, z)                      # sₖ = Azₖ
      MisI || mulorldiv!(w, M, s, ldiv)  # wₖ = M⁻¹sₖ
      @kaxpy!(n, -α, w, r)               # rₖ₊₁ = rₖ - αₖ * M⁻¹AN⁻¹(uₖ + qₖ)
      ρ_next = @kdot(n, c, r)            # ρₖ₊₁ = ⟨ r̅₀,rₖ₊₁ ⟩
      β = ρ_next / ρ                     # βₖ = ρₖ₊₁ / ρₖ
      @kcopy!(n, u, r)                   # uₖ₊₁ = rₖ₊₁
      @kaxpy!(n, β, q, u)                # uₖ₊₁ = uₖ₊₁ + βₖ * qₖ
      @kaxpby!(n, one(FC), q, β, p)      # pₐᵤₓ = qₖ + βₖ * pₖ
      @kaxpby!(n, one(FC), u, β, p)      # pₖ₊₁ = uₖ₊₁ + βₖ * pₐᵤₓ

      # Update ρ.
      ρ = ρ_next # ρₖ ← ρₖ₊₁

      # Update iteration index.
      iter = iter + 1

      # Compute residual norm ‖rₖ‖₂.
      rNorm = @knrm2(n, r)
      history && push!(rNorms, rNorm)

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      breakdown = (α == 0 || isnan(α))
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "breakdown αₖ == 0")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && @kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.storage = sizeof(solver)
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
