# An implementation of TriMR for the solution of symmetric and quasi-definite systems.
#
# This method is described in
#
# A. Montoison and D. Orban
# TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems.
# SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, June 2020.

export trimr, trimr!

"""
    (x, y, stats) = trimr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                          M=I, N=I, ldiv::Bool=false,
                          spd::Bool=false, snd::Bool=false,
                          flip::Bool=false, sp::Bool=false,
                          τ::T=one(T), ν::T=-one(T), atol::T=√eps(T),
                          rtol::T=√eps(T), itmax::Int=0,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = trimr(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

TriMR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Given a matrix `A` of dimension m × n, TriMR solves the symmetric linear system

    [ τE    A ] [ x ] = [ b ]
    [  Aᴴ  νF ] [ y ]   [ c ],

of size (n+m) × (n+m) where τ and ν are real numbers, E = M⁻¹ ≻ 0, F = N⁻¹ ≻ 0.
`b` and `c` must both be nonzero.
TriMR handles saddle-point systems (`τ = 0` or `ν = 0`) and adjoint systems (`τ = 0` and `ν = 0`) without any risk of breakdown.

By default, TriMR solves symmetric and quasi-definite linear systems with τ = 1 and ν = -1.

TriMR is based on the preconditioned orthogonal tridiagonalization process
and its relation with the preconditioned block-Lanczos process.

    [ M   0 ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

TriMR stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m;
* `c`: a vector of length n.

#### Optional arguments

* `x0`: a vector of length m that represents an initial guess of the solution x;
* `y0`: a vector of length n that represents an initial guess of the solution y.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the partitioned system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the partitioned system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `spd`: if `true`, set `τ = 1` and `ν = 1` for Hermitian and positive-definite linear system;
* `snd`: if `true`, set `τ = -1` and `ν = -1` for Hermitian and negative-definite linear systems;
* `flip`: if `true`, set `τ = -1` and `ν = 1` for another known variant of Hermitian quasi-definite systems;
* `sp`: if `true`, set `τ = 1` and `ν = 0` for saddle-point systems;
* `τ` and `ν`: diagonal scaling factors of the partitioned Hermitian linear system;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length m;
* `y`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* A. Montoison and D. Orban, [*TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems*](https://doi.org/10.1137/20M1363030), SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
"""
function trimr end

"""
    solver = trimr!(solver::TrimrSolver, A, b, c; kwargs...)
    solver = trimr!(solver::TrimrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`trimr`](@ref).

See [`TrimrSolver`](@ref) for more details about the `solver`.
"""
function trimr! end

def_args_trimr = (:(A                    ),
                  :(b::AbstractVector{FC}),
                  :(c::AbstractVector{FC}))

def_optargs_trimr = (:(x0::AbstractVector),
                     :(y0::AbstractVector))

def_kwargs_trimr = (:(; M = I                     ),
                    :(; N = I                     ),
                    :(; ldiv::Bool = false        ),
                    :(; spd::Bool = false         ),
                    :(; snd::Bool = false         ),
                    :(; flip::Bool = false        ),
                    :(; sp::Bool = false          ),
                    :(; τ::T = one(T)             ),
                    :(; ν::T = -one(T)            ),
                    :(; atol::T = √eps(T)         ),
                    :(; rtol::T = √eps(T)         ),
                    :(; itmax::Int = 0            ),
                    :(; timemax::Float64 = Inf    ),
                    :(; verbose::Int = 0          ),
                    :(; history::Bool = false     ),
                    :(; callback = solver -> false),
                    :(; iostream::IO = kstdout    ))

def_kwargs_trimr = mapreduce(extract_parameters, vcat, def_kwargs_trimr)

args_trimr = (:A, :b, :c)
optargs_trimr = (:x0, :y0)
kwargs_trimr = (:M, :N, :ldiv, :spd, :snd, :flip, :sp, :τ, :ν, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function trimr($(def_args_trimr...), $(def_optargs_trimr...); $(def_kwargs_trimr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = TrimrSolver(A, b)
    warm_start!(solver, $(optargs_trimr...))
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    trimr!(solver, $(args_trimr...); $(kwargs_trimr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.y, solver.stats)
  end

  function trimr($(def_args_trimr...); $(def_kwargs_trimr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}}
    start_time = time_ns()
    solver = TrimrSolver(A, b)
    elapsed_time = ktimer(start_time)
    timemax -= elapsed_time
    trimr!(solver, $(args_trimr...); $(kwargs_trimr...))
    solver.stats.timer += elapsed_time
    return (solver.x, solver.y, solver.stats)
  end

  function trimr!(solver :: TrimrSolver{T,FC,S}, $(def_args_trimr...); $(def_kwargs_trimr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "TriMR: system of %d equations in %d variables\n", m+n, m+n)

    # Check flip, sp, spd and snd parameters
    spd && flip && error("The matrix cannot be symmetric positive definite and symmetric quasi-definite !")
    spd && snd  && error("The matrix cannot be symmetric positive definite and symmetric negative definite !")
    spd && sp   && error("The matrix cannot be symmetric positive definite and a saddle-point !")
    snd && flip && error("The matrix cannot be symmetric negative definite and symmetric quasi-definite !")
    snd && sp   && error("The matrix cannot be symmetric negative definite and a saddle-point !")
    sp  && flip && error("The matrix cannot be symmetric quasi-definite and a saddle-point !")

    # Check M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Determine τ and ν associated to SQD, SPD or SND systems.
    flip && (τ = -one(T) ; ν =  one(T))
    spd  && (τ =  one(T) ; ν =  one(T))
    snd  && (τ = -one(T) ; ν = -one(T))
    sp   && (τ =  one(T) ; ν = zero(T))

    warm_start = solver.warm_start
    warm_start && (τ ≠ 0) && !MisI && error("Warm-start with preconditioners is not supported.")
    warm_start && (ν ≠ 0) && !NisI && error("Warm-start with preconditioners is not supported.")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :vₖ, S, m)
    allocate_if(!NisI, solver, :uₖ, S, n)
    Δy, yₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, p = solver.Δy, solver.y, solver.N⁻¹uₖ₋₁, solver.N⁻¹uₖ, solver.p
    Δx, xₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, q = solver.Δx, solver.x, solver.M⁻¹vₖ₋₁, solver.M⁻¹vₖ, solver.q
    gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ = solver.gy₂ₖ₋₃, solver.gy₂ₖ₋₂, solver.gy₂ₖ₋₁, solver.gy₂ₖ
    gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ = solver.gx₂ₖ₋₃, solver.gx₂ₖ₋₂, solver.gx₂ₖ₋₁, solver.gx₂ₖ
    vₖ = MisI ? M⁻¹vₖ : solver.vₖ
    uₖ = NisI ? N⁻¹uₖ : solver.uₖ
    vₖ₊₁ = MisI ? q : M⁻¹vₖ₋₁
    uₖ₊₁ = NisI ? p : N⁻¹uₖ₋₁
    b₀ = warm_start ? q : b
    c₀ = warm_start ? p : c

    stats = solver.stats
    rNorms = stats.residuals
    reset!(stats)

    # Initial solutions x₀ and y₀.
    xₖ .= zero(FC)
    yₖ .= zero(FC)

    iter = 0
    itmax == 0 && (itmax = m+n)

    # Initialize preconditioned orthogonal tridiagonalization process.
    M⁻¹vₖ₋₁ .= zero(FC)  # v₀ = 0
    N⁻¹uₖ₋₁ .= zero(FC)  # u₀ = 0

    # [ τI    A ] [ xₖ ] = [ b -  τΔx - AΔy ] = [ b₀ ]
    # [  Aᴴ  νI ] [ yₖ ]   [ c - AᴴΔx - νΔy ]   [ c₀ ]
    if warm_start
      mul!(b₀, A, Δy)
      (τ ≠ 0) && @kaxpy!(m, τ, Δx, b₀)
      @kaxpby!(m, one(FC), b, -one(FC), b₀)
      mul!(c₀, Aᴴ, Δx)
      (ν ≠ 0) && @kaxpy!(n, ν, Δy, c₀)
      @kaxpby!(n, one(FC), c, -one(FC), c₀)
    end

    # β₁Ev₁ = b ↔ β₁v₁ = Mb
    M⁻¹vₖ .= b₀
    MisI || mulorldiv!(vₖ, M, M⁻¹vₖ, ldiv)
    βₖ = sqrt(@kdotr(m, vₖ, M⁻¹vₖ))  # β₁ = ‖v₁‖_E
    if βₖ ≠ 0
      @kscal!(m, one(FC) / βₖ, M⁻¹vₖ)
      MisI || @kscal!(m, one(FC) / βₖ, vₖ)
    else
      error("b must be nonzero")
    end

    # γ₁Fu₁ = c ↔ γ₁u₁ = Nc
    N⁻¹uₖ .= c₀
    NisI || mulorldiv!(uₖ, N, N⁻¹uₖ, ldiv)
    γₖ = sqrt(@kdotr(n, uₖ, N⁻¹uₖ))  # γ₁ = ‖u₁‖_F
    if γₖ ≠ 0
      @kscal!(n, one(FC) / γₖ, N⁻¹uₖ)
      NisI || @kscal!(n, one(FC) / γₖ, uₖ)
    else
      error("c must be nonzero")
    end

    # Initialize directions Gₖ such that (GₖRₖ)ᵀ = (Wₖ)ᵀ.
    gx₂ₖ₋₃ .= zero(FC)
    gy₂ₖ₋₃ .= zero(FC)
    gx₂ₖ₋₂ .= zero(FC)
    gy₂ₖ₋₂ .= zero(FC)
    gx₂ₖ₋₁ .= zero(FC)
    gy₂ₖ₋₁ .= zero(FC)
    gx₂ₖ   .= zero(FC)
    gy₂ₖ   .= zero(FC)

    # Compute ‖r₀‖² = (γ₁)² + (β₁)²
    rNorm = sqrt(γₖ^2 + βₖ^2)
    history && push!(rNorms, rNorm)
    ε = atol + rtol * rNorm

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %5s\n", "k", "‖rₖ‖", "βₖ₊₁", "γₖ₊₁", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, βₖ, γₖ, ktimer(start_time))

    # Set up workspace.
    old_c₁ₖ = old_c₂ₖ = old_c₃ₖ = old_c₄ₖ = zero(T)
    old_s₁ₖ = old_s₂ₖ = old_s₃ₖ = old_s₄ₖ = zero(FC)
    σbar₂ₖ₋₂ = ηbar₂ₖ₋₃ = λbar₂ₖ₋₃ = μ₂ₖ₋₅ = λ₂ₖ₋₄ = μ₂ₖ₋₄ = zero(FC)
    πbar₂ₖ₋₁ = βₖ
    πbar₂ₖ = γₖ

    # Tolerance for breakdown detection.
    btol = eps(T)^(3/4)

    # Stopping criterion.
    breakdown = false
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    θbarₖ = δbar₂ₖ₋₁ = δbar₂ₖ = σbar₂ₖ₋₁ = σbar₂ₖ = λbar₂ₖ₋₁ = ηbar₂ₖ₋₁ = zero(FC)

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the orthogonal tridiagonalization process.
      # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = FUₖ(Tₖ)ᴴ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms Evₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms Fuₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        @kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q)  # q ← q - γₖ * M⁻¹vₖ₋₁
        @kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p)  # p ← p - βₖ * N⁻¹uₖ₋₁
      end

      αₖ = @kdot(m, vₖ, q)  # αₖ = ⟨vₖ,q⟩

      @kaxpy!(m, -     αₖ , M⁻¹vₖ, q)  # q ← q - αₖ * M⁻¹vₖ
      @kaxpy!(n, -conj(αₖ), N⁻¹uₖ, p)  # p ← p - ᾱₖ * N⁻¹uₖ

      # Compute vₖ₊₁ and uₖ₊₁
      MisI || mulorldiv!(vₖ₊₁, M, q, ldiv)  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
      NisI || mulorldiv!(uₖ₊₁, N, p, ldiv)  # γₖ₊₁uₖ₊₁ = NAᴴvₖ - βₖuₖ₋₁ - ᾱₖuₖ

      βₖ₊₁ = sqrt(@kdotr(m, vₖ₊₁, q))  # βₖ₊₁ = ‖vₖ₊₁‖_E
      γₖ₊₁ = sqrt(@kdotr(n, uₖ₊₁, p))  # γₖ₊₁ = ‖uₖ₊₁‖_F

      # βₖ₊₁ ≠ 0
      if βₖ₊₁ > btol
        @kscal!(m, one(FC) / βₖ₊₁, q)
        MisI || @kscal!(m, one(FC) / βₖ₊₁, vₖ₊₁)
      end

      # γₖ₊₁ ≠ 0
      if γₖ₊₁ > btol
        @kscal!(n, one(FC) / γₖ₊₁, p)
        NisI || @kscal!(n, one(FC) / γₖ₊₁, uₖ₊₁)
      end

      # Notations : Wₖ = [w₁ ••• wₖ] = [v₁ 0  ••• vₖ 0 ]
      #                                [0  u₁ ••• 0  uₖ]
      #
      # rₖ = [ b ] - [ τE    A ] [ xₖ ] = [ b ] - [ τE    A ] Wₖzₖ
      #      [ c ]   [  Aᴴ  νF ] [ yₖ ]   [ c ]   [  Aᴴ  νF ]
      #
      # block-Lanczos formulation : [ τE    A ] Wₖ = [ E   0 ] Wₖ₊₁Sₖ₊₁.ₖ
      #                             [  Aᴴ  νF ]      [ 0   F ]
      #
      # TriMR subproblem : min ‖ rₖ ‖ ↔ min ‖ Sₖ₊₁.ₖzₖ - β₁e₁ - γ₁e₂ ‖
      #
      # Update the QR factorization of Sₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      if iter == 1
        θbarₖ    = conj(αₖ)
        δbar₂ₖ₋₁ = τ
        δbar₂ₖ   = ν
        σbar₂ₖ₋₁ = αₖ
        σbar₂ₖ   = βₖ₊₁
        λbar₂ₖ₋₁ = γₖ₊₁
        ηbar₂ₖ₋₁ = zero(FC)
      else
        # Apply previous reflections
        #        [ 1                    ][ 1                    ][ c₂.ₖ₋₁  s₂.ₖ₋₁       ][ 1                    ]
        # Ζₖ₋₁ = [    c₄.ₖ₋₁  s₄.ₖ₋₁    ][    c₃.ₖ₋₁     s₃.ₖ₋₁ ][ s̄₂.ₖ₋₁ -c₂.ₖ₋₁       ][    c₁.ₖ₋₁     s₁.ₖ₋₁ ]
        #        [    s̄₄.ₖ₋₁ -c₄.ₖ₋₁    ][            1         ][                 1    ][            1         ]
        #        [                    1 ][    s̄₃.ₖ₋₁    -c₃.ₖ₋₁ ][                    1 ][    s̄₁.ₖ₋₁    -c₁.ₖ₋₁ ]
        #
        #        [ δbar₂ₖ₋₃  σbar₂ₖ₋₃ ηbar₂ₖ₋₃ λbar₂ₖ₋₃  0      0  ]   [ δ₂ₖ₋₃   σ₂ₖ₋₃  η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃       0      ]
        # Ζₖ₋₁ * [ θbarₖ₋₁   δbar₂ₖ₋₂ σbar₂ₖ₋₂    0      0      0  ] = [  0      δ₂ₖ₋₂  σ₂ₖ₋₂     η₂ₖ₋₂     λ₂ₖ₋₂     μ₂ₖ₋₂    ]
        #        [    0         βₖ       τ        αₖ     0    γₖ₊₁ ]   [  0        0    δbar₂ₖ₋₁  σbar₂ₖ₋₁  ηbar₂ₖ₋₁  λbar₂ₖ₋₁ ]
        #        [    γₖ        0        ᾱₖ       ν     βₖ₊₁    0  ]   [  0        0    θbarₖ     δbar₂ₖ    σbar₂ₖ      0      ]
        #
        # [ 1                    ] [ ηbar₂ₖ₋₃ λbar₂ₖ₋₃  0      0  ]   [ ηbar₂ₖ₋₃  λbar₂ₖ₋₃    0        0   ]
        # [    c₁.ₖ₋₁     s₁.ₖ₋₁ ] [ σbar₂ₖ₋₂    0      0      0  ] = [ σbis₂ₖ₋₂  ηbis₂ₖ₋₂  λbis₂ₖ₋₂   0   ]
        # [            1         ] [    τ        αₖ     0    γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
        # [    s̄₁.ₖ₋₁    -c₁.ₖ₋₁ ] [    ᾱₖ       ν     βₖ₊₁    0  ]   [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]
        σbis₂ₖ₋₂ =      old_c₁ₖ  * σbar₂ₖ₋₂ + old_s₁ₖ * conj(αₖ)
        ηbis₂ₖ₋₂ =                            old_s₁ₖ * ν
        λbis₂ₖ₋₂ =                            old_s₁ₖ * βₖ₊₁
        θbisₖ    = conj(old_s₁ₖ) * σbar₂ₖ₋₂ - old_c₁ₖ * conj(αₖ)
        δbis₂ₖ   =                          - old_c₁ₖ * ν
        σbis₂ₖ   =                          - old_c₁ₖ * βₖ₊₁
        # [ c₂.ₖ₋₁  s₂.ₖ₋₁       ] [ ηbar₂ₖ₋₃  λbar₂ₖ₋₃    0        0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]
        # [ s̄₂.ₖ₋₁ -c₂.ₖ₋₁       ] [ σbis₂ₖ₋₂  ηbis₂ₖ₋₂  λbis₂ₖ₋₂   0   ] = [ σhat₂ₖ₋₂  ηhat₂ₖ₋₂  λhat₂ₖ₋₂   0   ]
        # [                 1    ] [   τ        αₖ         0       γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
        # [                    1 ] [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]   [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]
        η₂ₖ₋₃    =      old_c₂ₖ  * ηbar₂ₖ₋₃ + old_s₂ₖ * σbis₂ₖ₋₂
        λ₂ₖ₋₃    =      old_c₂ₖ  * λbar₂ₖ₋₃ + old_s₂ₖ * ηbis₂ₖ₋₂
        μ₂ₖ₋₃    =                            old_s₂ₖ * λbis₂ₖ₋₂
        σhat₂ₖ₋₂ = conj(old_s₂ₖ) * ηbar₂ₖ₋₃ - old_c₂ₖ * σbis₂ₖ₋₂
        ηhat₂ₖ₋₂ = conj(old_s₂ₖ) * λbar₂ₖ₋₃ - old_c₂ₖ * ηbis₂ₖ₋₂
        λhat₂ₖ₋₂ =                          - old_c₂ₖ * λbis₂ₖ₋₂
        # [ 1                    ] [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]
        # [    c₃.ₖ₋₁     s₃.ₖ₋₁ ] [ σhat₂ₖ₋₂  ηhat₂ₖ₋₂  λhat₂ₖ₋₂   0   ] = [ σtmp₂ₖ₋₂  ηtmp₂ₖ₋₂  λtmp₂ₖ₋₂   0   ]
        # [            1         ] [   τ        αₖ         0       γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
        # [    s̄₃.ₖ₋₁    -c₃.ₖ₋₁ ] [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]   [  θbarₖ    δbar₂ₖ    σbar₂ₖ     0   ]
        σtmp₂ₖ₋₂ =      old_c₃ₖ  * σhat₂ₖ₋₂ + old_s₃ₖ * θbisₖ
        ηtmp₂ₖ₋₂ =      old_c₃ₖ  * ηhat₂ₖ₋₂ + old_s₃ₖ * δbis₂ₖ
        λtmp₂ₖ₋₂ =      old_c₃ₖ  * λhat₂ₖ₋₂ + old_s₃ₖ * σbis₂ₖ
        θbarₖ    = conj(old_s₃ₖ) * σhat₂ₖ₋₂ - old_c₃ₖ * θbisₖ
        δbar₂ₖ   = conj(old_s₃ₖ) * ηhat₂ₖ₋₂ - old_c₃ₖ * δbis₂ₖ
        σbar₂ₖ   = conj(old_s₃ₖ) * λhat₂ₖ₋₂ - old_c₃ₖ * σbis₂ₖ
        # [ 1                    ] [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃       0      ]
        # [    c₄.ₖ₋₁  s₄.ₖ₋₁    ] [ σtmp₂ₖ₋₂  ηtmp₂ₖ₋₂  λtmp₂ₖ₋₂   0   ] = [ σ₂ₖ₋₂     η₂ₖ₋₂     λ₂ₖ₋₂     μ₂ₖ₋₂    ]
        # [    s̄₄.ₖ₋₁ -c₄.ₖ₋₁    ] [   τ        αₖ         0       γₖ₊₁ ]   [ δbar₂ₖ₋₁  σbar₂ₖ₋₁  ηbar₂ₖ₋₁  λbar₂ₖ₋₁ ]
        # [                    1 ] [  θbarₖ    δbar₂ₖ    σbar₂ₖ     0   ]   [ θbarₖ     δbar₂ₖ    σbar₂ₖ      0      ]
        σ₂ₖ₋₂    =      old_c₄ₖ  * σtmp₂ₖ₋₂ + old_s₄ₖ * τ
        η₂ₖ₋₂    =      old_c₄ₖ  * ηtmp₂ₖ₋₂ + old_s₄ₖ * αₖ
        λ₂ₖ₋₂    =      old_c₄ₖ  * λtmp₂ₖ₋₂
        μ₂ₖ₋₂    =                            old_s₄ₖ * γₖ₊₁
        δbar₂ₖ₋₁ = conj(old_s₄ₖ) * σtmp₂ₖ₋₂ - old_c₄ₖ * τ
        σbar₂ₖ₋₁ = conj(old_s₄ₖ) * ηtmp₂ₖ₋₂ - old_c₄ₖ * αₖ
        ηbar₂ₖ₋₁ = conj(old_s₄ₖ) * λtmp₂ₖ₋₂
        λbar₂ₖ₋₁ =                          - old_c₄ₖ * γₖ₊₁
      end

      # [ 1                ] [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]   [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]
      # [    c₁.ₖ     s₁.ₖ ] [  θbarₖ     δbar₂ₖ  ] = [   θₖ       δbar₂ₖ  ]
      # [          1       ] [   0         βₖ₊₁   ]   [   0         βₖ₊₁   ]
      # [    s̄₁.ₖ    -c₁.ₖ ] [  γₖ₊₁        0     ]   [   0         gₖ     ]
      (c₁ₖ, s₁ₖ, θₖ) = sym_givens(θbarₖ, γₖ₊₁)
      gₖ     = conj(s₁ₖ) * δbar₂ₖ
      δbar₂ₖ =      c₁ₖ  * δbar₂ₖ

      # [ c₂.ₖ  s₂.ₖ       ] [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]
      # [ s̄₂.ₖ -c₂.ₖ       ] [   θₖ       δbar₂ₖ  ] = [  0     δbis₂ₖ ]
      # [             1    ] [   0         βₖ₊₁   ]   [  0      βₖ₊₁  ]
      # [                1 ] [   0         gₖ     ]   [  0       gₖ   ]
      (c₂ₖ, s₂ₖ, δ₂ₖ₋₁) = sym_givens(δbar₂ₖ₋₁, θₖ)
      σ₂ₖ₋₁  =      c₂ₖ  * σbar₂ₖ₋₁ + s₂ₖ * δbar₂ₖ
      δbis₂ₖ = conj(s₂ₖ) * σbar₂ₖ₋₁ - c₂ₖ * δbar₂ₖ

      # [ 1                ] [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]
      # [    c₃.ₖ     s₃.ₖ ] [  0     δbis₂ₖ ] = [  0     δhat₂ₖ ]
      # [          1       ] [  0      βₖ₊₁  ]   [  0      βₖ₊₁  ]
      # [    s̄₃.ₖ    -c₃.ₖ ] [  0       gₖ   ]   [  0       0    ]
      (c₃ₖ, s₃ₖ, δhat₂ₖ) = sym_givens(δbis₂ₖ, gₖ)

      # [ 1                ] [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁ ]
      # [    c₄.ₖ  s₄.ₖ    ] [  0     δhat₂ₖ ] = [  0      δ₂ₖ  ]
      # [    s̄₄.ₖ -c₄.ₖ    ] [  0      βₖ₊₁  ]   [  0       0   ]
      # [                1 ] [  0       0    ]   [  0       0   ]
      (c₄ₖ, s₄ₖ, δ₂ₖ) = sym_givens(δhat₂ₖ, βₖ₊₁)

      # Solve Gₖ = Wₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Gₖ)ᵀ = (Wₖ)ᵀ.
      if iter == 1
        # [ δ₁  0  ] [ gx₁ gy₁ ] = [ v₁ 0  ]
        # [ σ₁  δ₂ ] [ gx₂ gy₂ ]   [ 0  u₁ ]
        gx₂ₖ₋₁ .= vₖ ./ δ₂ₖ₋₁
        gx₂ₖ .= -(σ₂ₖ₋₁ / δ₂ₖ) .* gx₂ₖ₋₁
        gy₂ₖ .= uₖ ./ δ₂ₖ
      elseif iter == 2
        # [ η₁ σ₂ δ₃ 0  ] [ gx₁ gy₁ ] = [ v₂ 0  ]
        # [ λ₁ η₂ σ₃ δ₄ ] [ gx₂ gy₂ ]   [ 0  u₂ ]
        #                 [ gx₃ gy₃ ]
        #                 [ gx₄ gy₄ ]
        @kswap(gx₂ₖ₋₃, gx₂ₖ₋₁)
        @kswap(gx₂ₖ₋₂, gx₂ₖ)
        @kswap(gy₂ₖ₋₂, gy₂ₖ)
        gx₂ₖ₋₁ .= (vₖ .- η₂ₖ₋₃ .* gx₂ₖ₋₃ .- σ₂ₖ₋₂ .* gx₂ₖ₋₂                   ) ./ δ₂ₖ₋₁
        gx₂ₖ   .= (   .- λ₂ₖ₋₃ .* gx₂ₖ₋₃ .- η₂ₖ₋₂ .* gx₂ₖ₋₂ .- σ₂ₖ₋₁ .* gx₂ₖ₋₁) ./ δ₂ₖ
        gy₂ₖ₋₁ .= (   .- η₂ₖ₋₃ .* gy₂ₖ₋₃ .- σ₂ₖ₋₂ .* gy₂ₖ₋₂                   ) ./ δ₂ₖ₋₁
        gy₂ₖ   .= (uₖ .- λ₂ₖ₋₃ .* gy₂ₖ₋₃ .- η₂ₖ₋₂ .* gy₂ₖ₋₂ .- σ₂ₖ₋₁ .* gy₂ₖ₋₁) ./ δ₂ₖ
      else
        # μ₂ₖ₋₅ * gx₂ₖ₋₅ + λ₂ₖ₋₄ * gx₂ₖ₋₄ + η₂ₖ₋₃ * gx₂ₖ₋₃ + σ₂ₖ₋₂ * gx₂ₖ₋₂ + δ₂ₖ₋₁ * gx₂ₖ₋₁              = vₖ
        #                  μ₂ₖ₋₄ * gx₂ₖ₋₄ + λ₂ₖ₋₃ * gx₂ₖ₋₃ + η₂ₖ₋₂ * gx₂ₖ₋₂ + σ₂ₖ₋₁ * gx₂ₖ₋₁ + δ₂ₖ * gx₂ₖ = 0
        g₂ₖ₋₁ = g₂ₖ₋₅ = gx₂ₖ₋₃; g₂ₖ = g₂ₖ₋₄ = gx₂ₖ₋₂; g₂ₖ₋₃ = gx₂ₖ₋₁; g₂ₖ₋₂ = gx₂ₖ
        g₂ₖ₋₁ .= (vₖ .- μ₂ₖ₋₅ .* g₂ₖ₋₅ .- λ₂ₖ₋₄ .* g₂ₖ₋₄ .- η₂ₖ₋₃ .* g₂ₖ₋₃ .- σ₂ₖ₋₂ .* g₂ₖ₋₂                  ) ./ δ₂ₖ₋₁
        g₂ₖ   .= (                     .- μ₂ₖ₋₄ .* g₂ₖ₋₄ .- λ₂ₖ₋₃ .* g₂ₖ₋₃ .- η₂ₖ₋₂ .* g₂ₖ₋₂ .- σ₂ₖ₋₁ .* g₂ₖ₋₁) ./ δ₂ₖ
        @kswap(gx₂ₖ₋₃, gx₂ₖ₋₁)
        @kswap(gx₂ₖ₋₂, gx₂ₖ)
        # μ₂ₖ₋₅ * gy₂ₖ₋₅ + λ₂ₖ₋₄ * gy₂ₖ₋₄ + η₂ₖ₋₃ * gy₂ₖ₋₃ + σ₂ₖ₋₂ * gy₂ₖ₋₂ + δ₂ₖ₋₁ * gy₂ₖ₋₁              = 0
        #                  μ₂ₖ₋₄ * gy₂ₖ₋₄ + λ₂ₖ₋₃ * gy₂ₖ₋₃ + η₂ₖ₋₂ * gy₂ₖ₋₂ + σ₂ₖ₋₁ * gy₂ₖ₋₁ + δ₂ₖ * gy₂ₖ = uₖ
        g₂ₖ₋₁ = g₂ₖ₋₅ = gy₂ₖ₋₃; g₂ₖ = g₂ₖ₋₄ = gy₂ₖ₋₂; g₂ₖ₋₃ = gy₂ₖ₋₁; g₂ₖ₋₂ = gy₂ₖ
        g₂ₖ₋₁ .= (     .- μ₂ₖ₋₅ .* g₂ₖ₋₅ .- λ₂ₖ₋₄ .* g₂ₖ₋₄ .- η₂ₖ₋₃ .* g₂ₖ₋₃ .- σ₂ₖ₋₂ .* g₂ₖ₋₂                  ) ./ δ₂ₖ₋₁
        g₂ₖ   .= (uₖ                     .- μ₂ₖ₋₄ .* g₂ₖ₋₄ .- λ₂ₖ₋₃ .* g₂ₖ₋₃ .- η₂ₖ₋₂ .* g₂ₖ₋₂ .- σ₂ₖ₋₁ .* g₂ₖ₋₁) ./ δ₂ₖ
        @kswap(gy₂ₖ₋₃, gy₂ₖ₋₁)
        @kswap(gy₂ₖ₋₂, gy₂ₖ)
      end

      # Update p̅ₖ = (Qₖ)ᴴ * (β₁e₁ + γ₁e₂)
      πbis₂ₖ   =      c₁ₖ  * πbar₂ₖ
      πbis₂ₖ₊₂ = conj(s₁ₖ) * πbar₂ₖ
      #
      π₂ₖ₋₁  =      c₂ₖ  * πbar₂ₖ₋₁ + s₂ₖ * πbis₂ₖ
      πhat₂ₖ = conj(s₂ₖ) * πbar₂ₖ₋₁ - c₂ₖ * πbis₂ₖ
      #
      πtmp₂ₖ   =      c₃ₖ  * πhat₂ₖ + s₃ₖ * πbis₂ₖ₊₂
      πbar₂ₖ₊₂ = conj(s₃ₖ) * πhat₂ₖ - c₃ₖ * πbis₂ₖ₊₂
      #
      π₂ₖ      =      c₄ₖ  * πtmp₂ₖ
      πbar₂ₖ₊₁ = conj(s₄ₖ) * πtmp₂ₖ

      # Update xₖ = Gxₖ * pₖ
      @kaxpy!(m, π₂ₖ₋₁, gx₂ₖ₋₁, xₖ)
      @kaxpy!(m, π₂ₖ  , gx₂ₖ  , xₖ)

      # Update yₖ = Gyₖ * pₖ
      @kaxpy!(n, π₂ₖ₋₁, gy₂ₖ₋₁, yₖ)
      @kaxpy!(n, π₂ₖ  , gy₂ₖ  , yₖ)

      # Compute ‖rₖ‖² = |πbar₂ₖ₊₁|² + |πbar₂ₖ₊₂|²
      rNorm = sqrt(abs2(πbar₂ₖ₊₁) + abs2(πbar₂ₖ₊₂))
      history && push!(rNorms, rNorm)

      # Update vₖ and uₖ
      MisI || (vₖ .= vₖ₊₁)
      NisI || (uₖ .= uₖ₊₁)

      # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
      M⁻¹vₖ₋₁ .= M⁻¹vₖ
      N⁻¹uₖ₋₁ .= N⁻¹uₖ

      # Update M⁻¹vₖ and N⁻¹uₖ
      M⁻¹vₖ .= q
      N⁻¹uₖ .= p

      # Update cosines and sines
      old_s₁ₖ = s₁ₖ
      old_s₂ₖ = s₂ₖ
      old_s₃ₖ = s₃ₖ
      old_s₄ₖ = s₄ₖ
      old_c₁ₖ = c₁ₖ
      old_c₂ₖ = c₂ₖ
      old_c₃ₖ = c₃ₖ
      old_c₄ₖ = c₄ₖ

      # Update workspace
      βₖ = βₖ₊₁
      γₖ = γₖ₊₁
      σbar₂ₖ₋₂ = σbar₂ₖ
      ηbar₂ₖ₋₃ = ηbar₂ₖ₋₁
      λbar₂ₖ₋₃ = λbar₂ₖ₋₁
      if iter ≥ 2
        μ₂ₖ₋₅ = μ₂ₖ₋₃
        μ₂ₖ₋₄ = μ₂ₖ₋₂
        λ₂ₖ₋₄ = λ₂ₖ₋₂
      end
      πbar₂ₖ₋₁ = πbar₂ₖ₊₁
      πbar₂ₖ   = πbar₂ₖ₊₂

      # Stopping conditions that do not depend on user input.
      # This is to guard against tolerances that are unreasonably small.
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      breakdown = βₖ₊₁ ≤ btol && γₖ₊₁ ≤ btol
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, βₖ₊₁, γₖ₊₁, ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "inconsistent linear system")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x and y
    warm_start && @kaxpy!(m, one(FC), Δx, xₖ)
    warm_start && @kaxpy!(n, one(FC), Δy, yₖ)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = !solved && breakdown
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
