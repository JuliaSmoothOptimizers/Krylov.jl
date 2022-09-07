# An implementation of TriCG for the solution of symmetric and quasi-definite systems.
#
# This method is described in
#
# A. Montoison and D. Orban
# TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems.
# SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, April 2020.

export tricg, tricg!

"""
    (x, y, stats) = tricg(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                          M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                          spd::Bool=false, snd::Bool=false, flip::Bool=false,
                          τ::T=one(T), ν::T=-one(T), itmax::Int=0,
                          verbose::Int=0, history::Bool=false,
                          ldiv::Bool=false, callback=solver->false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

TriCG solves the symmetric linear system

    [ τE    A ] [ x ] = [ b ]
    [  Aᴴ  νF ] [ y ]   [ c ],

where τ and ν are real numbers, E = M⁻¹ ≻ 0 and F = N⁻¹ ≻ 0.
`b` and `c` must both be nonzero.
TriCG could breakdown if `τ = 0` or `ν = 0`.
It's recommended to use TriMR in these cases.

By default, TriCG solves symmetric and quasi-definite linear systems with τ = 1 and ν = -1.
If `flip = true`, TriCG solves another known variant of SQD systems where τ = -1 and ν = 1.
If `spd = true`, τ = ν = 1 and the associated symmetric and positive definite linear system is solved.
If `snd = true`, τ = ν = -1 and the associated symmetric and negative definite linear system is solved.
`τ` and `ν` are also keyword arguments that can be directly modified for more specific problems.

TriCG is based on the preconditioned orthogonal tridiagonalization process
and its relation with the preconditioned block-Lanczos process.

    [ M   0 ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

TriCG stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

TriCG can be warm-started from initial guesses `x0` and `y0` with the method

    (x, y, stats) = tricg(A, b, c, x0, y0; kwargs...)

where `kwargs` are the same keyword arguments as above.

The callback is called as `callback(solver)` and should return `true` if the main loop should terminate,
and `false` otherwise.

#### Reference

* A. Montoison and D. Orban, [*TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems*](https://doi.org/10.1137/20M1363030), SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
"""
function tricg end

function tricg(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = TricgSolver(A, b)
  tricg!(solver, A, b, c, x0, y0; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

function tricg(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = TricgSolver(A, b)
  tricg!(solver, A, b, c; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = tricg!(solver::TricgSolver, A, b, c; kwargs...)
    solver = tricg!(solver::TricgSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`tricg`](@ref).

See [`TricgSolver`](@ref) for more details about the `solver`.
"""
function tricg! end

function tricg!(solver :: TricgSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0, y0)
  tricg!(solver, A, b, c; kwargs...)
  return solver
end

function tricg!(solver :: TricgSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC};
                M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                spd :: Bool=false, snd :: Bool=false, flip :: Bool=false,
                τ :: T=one(T), ν :: T=-one(T), itmax :: Int=0,
                verbose :: Int=0, history :: Bool=false,
                ldiv :: Bool=false, callback = solver -> false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("TriCG: system of %d equations in %d variables\n", m+n, m+n)

  # Check flip, spd and snd parameters
  spd && flip && error("The matrix cannot be SPD and SQD")
  snd && flip && error("The matrix cannot be SND and SQD")
  spd && snd  && error("The matrix cannot be SPD and SND")

  # Check M = Iₘ and N = Iₙ
  MisI = (M === I)
  NisI = (N === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Determine τ and ν associated to SQD, SPD or SND systems.
  flip && (τ = -one(T) ; ν =  one(T))
  spd  && (τ =  one(T) ; ν =  one(T))
  snd  && (τ = -one(T) ; ν = -one(T))

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
  gy₂ₖ₋₁, gy₂ₖ, gx₂ₖ₋₁, gx₂ₖ = solver.gy₂ₖ₋₁, solver.gy₂ₖ, solver.gx₂ₖ₋₁, solver.gx₂ₖ
  vₖ = MisI ? M⁻¹vₖ : solver.vₖ
  uₖ = NisI ? N⁻¹uₖ : solver.uₖ
  vₖ₊₁ = MisI ? q : vₖ
  uₖ₊₁ = NisI ? p : uₖ
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

  # Initialize directions Gₖ such that L̄ₖ(Gₖ)ᵀ = (Wₖ)ᵀ
  gx₂ₖ₋₁ .= zero(FC)
  gy₂ₖ₋₁ .= zero(FC)
  gx₂ₖ   .= zero(FC)
  gy₂ₖ   .= zero(FC)

  # Compute ‖r₀‖² = (γ₁)² + (β₁)²
  rNorm = sqrt(γₖ^2 + βₖ^2)
  history && push!(rNorms, rNorm)
  ε = atol + rtol * rNorm

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s\n", "k", "‖rₖ‖", "βₖ₊₁", "γₖ₊₁")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e\n", iter, rNorm, βₖ, γₖ)

  # Set up workspace.
  d₂ₖ₋₃ = d₂ₖ₋₂ = zero(T)
  π₂ₖ₋₃ = π₂ₖ₋₂ = zero(FC)
  δₖ₋₁ = zero(FC)

  # Tolerance for breakdown detection.
  btol = eps(T)^(3/4)

  # Stopping criterion.
  breakdown = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"
  user_requested_exit = false

  while !(solved || tired || breakdown || user_requested_exit)
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

    # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
    M⁻¹vₖ₋₁ .= M⁻¹vₖ
    N⁻¹uₖ₋₁ .= N⁻¹uₖ

    # Notations : Wₖ = [w₁ ••• wₖ] = [v₁ 0  ••• vₖ 0 ]
    #                                [0  u₁ ••• 0  uₖ]
    #
    # rₖ = [ b ] - [ τE    A ] [ xₖ ] = [ b ] - [ τE    A ] Wₖzₖ
    #      [ c ]   [  Aᴴ  νF ] [ yₖ ]   [ c ]   [  Aᴴ  νF ]
    #
    # block-Lanczos formulation : [ τE    A ] Wₖ = [ E   0 ] Wₖ₊₁Sₖ₊₁.ₖ
    #                             [  Aᴴ  νF ]      [ 0   F ]
    #
    # TriCG subproblem : (Wₖ)ᴴ * rₖ = 0 ↔ Sₖ.ₖzₖ = β₁e₁ + γ₁e₂
    #
    # Update the LDLᴴ factorization of Sₖ.ₖ.
    #
    # [ τ  α₁    γ₂ 0  •  •  •  •  0  ]
    # [ ᾱ₁ ν  β₂       •           •  ]
    # [    β₂ τ  α₂    γ₃ •        •  ]
    # [ γ₂    ᾱ₂ ν  β₃       •     •  ]
    # [ 0        β₃ •  •     •  •  •  ]
    # [ •  •  γ₃    •  •  •        0  ]
    # [ •     •        •  •  •     γₖ ]
    # [ •        •  •     •  •  βₖ    ]
    # [ •           •        βₖ τ  αₖ ]
    # [ 0  •  •  •  •  0  γₖ    ᾱₖ ν  ]
    if iter == 1
      d₂ₖ₋₁ = τ
      δₖ    = conj(αₖ) / d₂ₖ₋₁
      d₂ₖ   = ν - abs2(δₖ) * d₂ₖ₋₁
    else
      σₖ    = βₖ / d₂ₖ₋₂
      ηₖ    = γₖ / d₂ₖ₋₃
      λₖ    = -(ηₖ * conj(δₖ₋₁) * d₂ₖ₋₃) / d₂ₖ₋₂
      d₂ₖ₋₁ = τ - abs2(σₖ) * d₂ₖ₋₂
      δₖ    = (conj(αₖ) - λₖ * conj(σₖ) * d₂ₖ₋₂) / d₂ₖ₋₁
      d₂ₖ   = ν - abs2(ηₖ) * d₂ₖ₋₃ - abs2(λₖ) * d₂ₖ₋₂ - abs2(δₖ) * d₂ₖ₋₁
    end

    # Solve LₖDₖpₖ = (β₁e₁ + γ₁e₂)
    #
    # [ 1  0  •  •  •  •  •  •  •  0 ] [ d₁                        ]      [ β₁ ]
    # [ δ₁ 1  •                    • ] [    d₂                     ]      [ γ₁ ]
    # [    σ₂ 1  •                 • ] [       •                   ]      [ 0  ]
    # [ η₂ λ₂ δ₂ 1  •              • ] [         •                 ]      [ •  ]
    # [ 0        σ₃ 1  •           • ] [           •               ] zₖ = [ •  ]
    # [ •  •  η₃ λ₃ δ₃ 1  •        • ] [             •             ]      [ •  ]
    # [ •     •        •  •  •     • ] [               •           ]      [ •  ]
    # [ •        •  •  •  •  •  •  • ] [                 •         ]      [ •  ]
    # [ •           •        σₖ 1  0 ] [                   d₂ₖ₋₁   ]      [ •  ]
    # [ 0  •  •  •  •  0  ηₖ λₖ δₖ 1 ] [                        d₂ₖ]      [ 0  ]
    if iter == 1
      π₂ₖ₋₁ = βₖ / d₂ₖ₋₁
      π₂ₖ   = (γₖ - δₖ * βₖ) / d₂ₖ
    else
      π₂ₖ₋₁ = -(σₖ * d₂ₖ₋₂ * π₂ₖ₋₂) / d₂ₖ₋₁
      π₂ₖ   = -(δₖ * d₂ₖ₋₁ * π₂ₖ₋₁ + λₖ * d₂ₖ₋₂ * π₂ₖ₋₂ + ηₖ * d₂ₖ₋₃ * π₂ₖ₋₃) / d₂ₖ
    end

    # Solve Gₖ = Wₖ(Lₖ)⁻ᴴ ⟷ L̄ₖ(Gₖ)ᵀ = (Wₖ)ᵀ.
    if iter == 1
      # [ 1  0 ] [ gx₁ gy₁ ] = [ v₁ 0  ]
      # [ δ̄₁ 1 ] [ gx₂ gy₂ ]   [ 0  u₁ ]
      @. gx₂ₖ₋₁ = vₖ
      @. gx₂ₖ   = - conj(δₖ) * gx₂ₖ₋₁
      @. gy₂ₖ   = uₖ
    else
      # [ 0  σ̄ₖ 1  0 ] [ gx₂ₖ₋₃ gy₂ₖ₋₃ ] = [ vₖ 0  ]
      # [ η̄ₖ λ̄ₖ δ̄ₖ 1 ] [ gx₂ₖ₋₂ gy₂ₖ₋₂ ]   [ 0  uₖ ]
      #                [ gx₂ₖ₋₁ gy₂ₖ₋₁ ]
      #                [ gx₂ₖ   gy₂ₖ   ]
      @. gx₂ₖ₋₁ = conj(ηₖ) * gx₂ₖ₋₁ + conj(λₖ) * gx₂ₖ
      @. gy₂ₖ₋₁ = conj(ηₖ) * gy₂ₖ₋₁ + conj(λₖ) * gy₂ₖ

      @. gx₂ₖ = vₖ - conj(σₖ) * gx₂ₖ
      @. gy₂ₖ =    - conj(σₖ) * gy₂ₖ

      @. gx₂ₖ₋₁ =    - gx₂ₖ₋₁ - conj(δₖ) * gx₂ₖ
      @. gy₂ₖ₋₁ = uₖ - gy₂ₖ₋₁ - conj(δₖ) * gy₂ₖ

      # g₂ₖ₋₃ == g₂ₖ and g₂ₖ₋₂ == g₂ₖ₋₁
      @kswap(gx₂ₖ₋₁, gx₂ₖ)
      @kswap(gy₂ₖ₋₁, gy₂ₖ)
    end

    # Update xₖ = Gxₖ * pₖ
    @kaxpy!(m, π₂ₖ₋₁, gx₂ₖ₋₁, xₖ)
    @kaxpy!(m, π₂ₖ  , gx₂ₖ  , xₖ)

    # Update yₖ = Gyₖ * pₖ
    @kaxpy!(n, π₂ₖ₋₁, gy₂ₖ₋₁, yₖ)
    @kaxpy!(n, π₂ₖ  , gy₂ₖ  , yₖ)

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

    # Update M⁻¹vₖ and N⁻¹uₖ
    M⁻¹vₖ .= q
    N⁻¹uₖ .= p

    # Compute ‖rₖ‖² = |γₖ₊₁ζ₂ₖ₋₁|² + |βₖ₊₁ζ₂ₖ|²
    ζ₂ₖ₋₁ = π₂ₖ₋₁ - conj(δₖ) * π₂ₖ
    ζ₂ₖ   = π₂ₖ
    rNorm = sqrt(abs2(γₖ₊₁ * ζ₂ₖ₋₁) + abs2(βₖ₊₁ * ζ₂ₖ))
    history && push!(rNorms, rNorm)

    # Update βₖ, γₖ, π₂ₖ₋₃, π₂ₖ₋₂, d₂ₖ₋₃, d₂ₖ₋₂, δₖ₋₁, vₖ, uₖ.
    βₖ    = βₖ₊₁
    γₖ    = γₖ₊₁
    π₂ₖ₋₃ = π₂ₖ₋₁
    π₂ₖ₋₂ = π₂ₖ
    d₂ₖ₋₃ = d₂ₖ₋₁
    d₂ₖ₋₂ = d₂ₖ
    δₖ₋₁  = δₖ

    # Stopping conditions that do not depend on user input.
    # This is to guard against tolerances that are unreasonably small.
    resid_decrease_mach = (rNorm + one(T) ≤ one(T))

    # Update stopping criterion.
    user_requested_exit = callback(solver) :: Bool
    resid_decrease_lim = rNorm ≤ ε
    breakdown = βₖ₊₁ ≤ btol && γₖ₊₁ ≤ btol
    solved = resid_decrease_lim || resid_decrease_mach
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e\n", iter, rNorm, βₖ₊₁, γₖ₊₁)
  end
  (verbose > 0) && @printf("\n")

  tired               && (status = "maximum number of iterations exceeded")
  breakdown           && (status = "inconsistent linear system")
  solved              && (status = "solution good enough given atol and rtol")
  user_requested_exit && (status = "user-requested exit")

  # Update x and y
  warm_start && @kaxpy!(m, one(FC), Δx, xₖ)
  warm_start && @kaxpy!(n, one(FC), Δy, yₖ)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = !solved && breakdown
  stats.status = status
  return solver
end
