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
                          M=I, N=I, ldiv::Bool=false,
                          spd::Bool=false, snd::Bool=false,
                          flip::Bool=false, τ::T=one(T),
                          ν::T=-one(T), atol::T=√eps(T),
                          rtol::T=√eps(T), itmax::Int=0,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = tricg(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

TriCG can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Given a matrix `A` of dimension m × n, TriCG solves the Hermitian linear system

    [ τE    A ] [ x ] = [ b ]
    [  Aᴴ  νF ] [ y ]   [ c ],

of size (n+m) × (n+m) where τ and ν are real numbers, E = M⁻¹ ≻ 0 and F = N⁻¹ ≻ 0.
TriCG could breakdown if `τ = 0` or `ν = 0`.
It's recommended to use TriMR in these cases.

By default, TriCG solves Hermitian and quasi-definite linear systems with τ = 1 and ν = -1.

TriCG is based on the preconditioned orthogonal tridiagonalization process
and its relation with the preconditioned block-Lanczos process.

    [ M   0 ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

TriCG stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

#### Interface

To easily switch between Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :tricg`.

For an in-place variant that reuses memory across solves, see [`tricg!`](@ref).

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`.

#### Optional arguments

* `x0`: a vector of length `m` that represents an initial guess of the solution `x`;
* `y0`: a vector of length `n` that represents an initial guess of the solution `y`.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the partitioned system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the partitioned system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `spd`: if `true`, set `τ = 1` and `ν = 1` for Hermitian and positive-definite linear system;
* `snd`: if `true`, set `τ = -1` and `ν = -1` for Hermitian and negative-definite linear systems;
* `flip`: if `true`, set `τ = -1` and `ν = 1` for another known variant of Hermitian quasi-definite systems;
* `τ` and `ν`: diagonal scaling factors of the partitioned Hermitian linear system;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `m`;
* `y`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* A. Montoison and D. Orban, [*TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems*](https://doi.org/10.1137/20M1363030), SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
"""
function tricg end

"""
    workspace = tricg!(workspace::TricgWorkspace, A, b, c; kwargs...)
    workspace = tricg!(workspace::TricgWorkspace, A, b, c, x0, y0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`tricg`](@ref).

See [`TricgWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) to allocate the workspace,
and [`krylov_solve!`](@ref) to run the Krylov method in-place.
"""
function tricg! end

def_args_tricg = (:(A                    ),
                  :(b::AbstractVector{FC}),
                  :(c::AbstractVector{FC}))

def_optargs_tricg = (:(x0::AbstractVector),
                     :(y0::AbstractVector))

def_kwargs_tricg = (:(; M = I                        ),
                    :(; N = I                        ),
                    :(; ldiv::Bool = false           ),
                    :(; spd::Bool = false            ),
                    :(; snd::Bool = false            ),
                    :(; flip::Bool = false           ),
                    :(; τ::T = one(T)                ),
                    :(; ν::T = -one(T)               ),
                    :(; atol::T = √eps(T)            ),
                    :(; rtol::T = √eps(T)            ),
                    :(; itmax::Int = 0               ),
                    :(; timemax::Float64 = Inf       ),
                    :(; verbose::Int = 0             ),
                    :(; history::Bool = false        ),
                    :(; callback = workspace -> false),
                    :(; iostream::IO = kstdout       ))

def_kwargs_tricg = extract_parameters.(def_kwargs_tricg)

args_tricg = (:A, :b, :c)
optargs_tricg = (:x0, :y0)
kwargs_tricg = (:M, :N, :ldiv, :spd, :snd, :flip, :τ, :ν, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function tricg!(workspace :: TricgWorkspace{T,FC,S}, $(def_args_tricg...); $(def_kwargs_tricg...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "TriCG: system of %d equations in %d variables\n", m+n, m+n)

    # Check flip, spd and snd parameters
    spd && flip && error("The matrix cannot be SPD and SQD")
    snd && flip && error("The matrix cannot be SND and SQD")
    spd && snd  && error("The matrix cannot be SPD and SND")

    # Check M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")
    ktypeof(c) == S || error("ktypeof(c) must be equal to $S")

    # Determine τ and ν associated to SQD, SPD or SND systems.
    flip && (τ = -one(T) ; ν =  one(T))
    spd  && (τ =  one(T) ; ν =  one(T))
    snd  && (τ = -one(T) ; ν = -one(T))

    warm_start = workspace.warm_start
    warm_start && (τ ≠ 0) && !MisI && error("Warm-start with preconditioners is not supported.")
    warm_start && (ν ≠ 0) && !NisI && error("Warm-start with preconditioners is not supported.")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, workspace, :vₖ, S, workspace.x)  # The length of vₖ is m
    allocate_if(!NisI, workspace, :uₖ, S, workspace.y)  # The length of uₖ is n
    Δy, yₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, p = workspace.Δy, workspace.y, workspace.N⁻¹uₖ₋₁, workspace.N⁻¹uₖ, workspace.p
    Δx, xₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, q = workspace.Δx, workspace.x, workspace.M⁻¹vₖ₋₁, workspace.M⁻¹vₖ, workspace.q
    gy₂ₖ₋₁, gy₂ₖ, gx₂ₖ₋₁, gx₂ₖ = workspace.gy₂ₖ₋₁, workspace.gy₂ₖ, workspace.gx₂ₖ₋₁, workspace.gx₂ₖ
    vₖ = MisI ? M⁻¹vₖ : workspace.vₖ
    uₖ = NisI ? N⁻¹uₖ : workspace.uₖ
    vₖ₊₁ = MisI ? q : vₖ
    uₖ₊₁ = NisI ? p : uₖ
    b₀ = warm_start ? q : b
    c₀ = warm_start ? p : c

    stats = workspace.stats
    rNorms = stats.residuals
    reset!(stats)

    # Initial solutions x₀ and y₀.
    kfill!(xₖ, zero(FC))
    kfill!(yₖ, zero(FC))

    iter = 0
    itmax == 0 && (itmax = m+n)

    # Initialize preconditioned orthogonal tridiagonalization process.
    kfill!(M⁻¹vₖ₋₁, zero(FC))  # v₀ = 0
    kfill!(N⁻¹uₖ₋₁, zero(FC))  # u₀ = 0

    # [ τI    A ] [ xₖ ] = [ b -  τΔx - AΔy ] = [ b₀ ]
    # [  Aᴴ  νI ] [ yₖ ]   [ c - AᴴΔx - νΔy ]   [ c₀ ]
    if warm_start
      mul!(b₀, A, Δy)
      (τ ≠ 0) && kaxpy!(m, τ, Δx, b₀)
      kaxpby!(m, one(FC), b, -one(FC), b₀)
      mul!(c₀, Aᴴ, Δx)
      (ν ≠ 0) && kaxpy!(n, ν, Δy, c₀)
      kaxpby!(n, one(FC), c, -one(FC), c₀)
    end

    # β₁Ev₁ = b ↔ β₁v₁ = Mb
    kcopy!(m, M⁻¹vₖ, b₀)  # M⁻¹vₖ ← b₀
    MisI || mulorldiv!(vₖ, M, M⁻¹vₖ, ldiv)
    βₖ = knorm_elliptic(m, vₖ, M⁻¹vₖ)  # β₁ = ‖v₁‖_E
    if βₖ ≠ 0
      kdiv!(m, M⁻¹vₖ, βₖ)
      MisI || kdiv!(m, vₖ, βₖ)
    else
      # v₁ = 0 such that v₁ ⊥ Span{v₁, ..., vₖ}
      kfill!(M⁻¹vₖ, zero(FC))
      MisI || kfill!(vₖ, zero(FC))
    end

    # γ₁Fu₁ = c ↔ γ₁u₁ = Nc
    kcopy!(n, N⁻¹uₖ, c₀)  # M⁻¹uₖ ← c₀
    NisI || mulorldiv!(uₖ, N, N⁻¹uₖ, ldiv)
    γₖ = knorm_elliptic(n, uₖ, N⁻¹uₖ)  # γ₁ = ‖u₁‖_F
    if γₖ ≠ 0
      kdiv!(n, N⁻¹uₖ, γₖ)
      NisI || kdiv!(n, uₖ, γₖ)
    else
      # u₁ = 0 such that u₁ ⊥ Span{u₁, ..., uₖ}
      kfill!(N⁻¹uₖ, zero(FC))
      NisI || kfill!(uₖ, zero(FC))
    end

    # Initialize directions Gₖ such that L̄ₖ(Gₖ)ᵀ = (Wₖ)ᵀ
    kfill!(gx₂ₖ₋₁, zero(FC))
    kfill!(gy₂ₖ₋₁, zero(FC))
    kfill!(gx₂ₖ  , zero(FC))
    kfill!(gy₂ₖ  , zero(FC))

    # Compute ‖r₀‖² = (γ₁)² + (β₁)²
    rNorm = sqrt(γₖ^2 + βₖ^2)
    history && push!(rNorms, rNorm)
    ε = atol + rtol * rNorm

    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %7s  %5s\n", "k", "‖rₖ‖", "βₖ₊₁", "γₖ₊₁", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, βₖ, γₖ, start_time |> ktimer)

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
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the orthogonal tridiagonalization process.
      # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = FUₖ(Tₖ)ᴴ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms Evₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms Fuₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q)  # q ← q - γₖ * M⁻¹vₖ₋₁
        kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p)  # p ← p - βₖ * N⁻¹uₖ₋₁
      end

      αₖ = kdot(m, vₖ, q)  # αₖ = ⟨vₖ,q⟩

      kaxpy!(m, -     αₖ , M⁻¹vₖ, q)  # q ← q - αₖ * M⁻¹vₖ
      kaxpy!(n, -conj(αₖ), N⁻¹uₖ, p)  # p ← p - ᾱₖ * N⁻¹uₖ

      # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
      kcopy!(m, M⁻¹vₖ₋₁, M⁻¹vₖ)  # M⁻¹vₖ₋₁ ← M⁻¹vₖ
      kcopy!(n, N⁻¹uₖ₋₁, N⁻¹uₖ)  # N⁻¹uₖ₋₁ ← N⁻¹uₖ

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
        kcopy!(m, gx₂ₖ₋₁, vₖ)                   # gx₂ₖ₋₁ = vₖ
        kscalcopy!(m, gx₂ₖ, -conj(δₖ), gx₂ₖ₋₁)  # gx₂ₖ = -conj(δₖ) * gx₂ₖ₋₁
        kcopy!(n, gy₂ₖ, uₖ)                     # gy₂ₖ = uₖ
      else
        # [ 0  σ̄ₖ 1  0 ] [ gx₂ₖ₋₃ gy₂ₖ₋₃ ] = [ vₖ 0  ]
        # [ η̄ₖ λ̄ₖ δ̄ₖ 1 ] [ gx₂ₖ₋₂ gy₂ₖ₋₂ ]   [ 0  uₖ ]
        #                [ gx₂ₖ₋₁ gy₂ₖ₋₁ ]
        #                [ gx₂ₖ   gy₂ₖ   ]
        #
        # gx₂ₖ₋₁ = vₖ - σ̄ₖ * gx₂ₖ₋₂
        # gy₂ₖ₋₁ =    - σ̄ₖ * gy₂ₖ₋₂
        # gx₂ₖ   =    - η̄ₖ * gx₂ₖ₋₃ - λ̄ₖ * gx₂ₖ₋₂ - δ̄ₖ * gx₂ₖ₋₁
        # gy₂ₖ   = uₖ - η̄ₖ * gy₂ₖ₋₃ - λ̄ₖ * gy₂ₖ₋₂ - δ̄ₖ * gy₂ₖ₋₁
        #
        # gx₂ₖ and gy₂ₖ will reuse the same storage as gx₂ₖ₋₃ and gy₂ₖ₋₃
        # gx₂ₖ₋₁ and gy₂ₖ₋₁ will reuse the same storage as gx₂ₖ₋₂ and gy₂ₖ₋₂

        kaxpby!(m, conj(λₖ), gx₂ₖ, conj(ηₖ), gx₂ₖ₋₁)  # gx₂ₖ₋₁ = conj(ηₖ) * gx₂ₖ₋₃ + conj(λₖ) * gx₂ₖ₋₂
        kaxpby!(n, conj(λₖ), gy₂ₖ, conj(ηₖ), gy₂ₖ₋₁)  # gy₂ₖ₋₁ = conj(ηₖ) * gy₂ₖ₋₃ + conj(λₖ) * gy₂ₖ₋₂

        kaxpby!(m, one(FC), vₖ, -conj(σₖ), gx₂ₖ)  # gx₂ₖ = vₖ - conj(σₖ) * gx₂ₖ₋₂
        kscal!(n, -conj(σₖ), gy₂ₖ)                # gy₂ₖ =    - conj(σₖ) * gy₂ₖ₋₂

        kaxpby!(m, -conj(δₖ), gx₂ₖ, -one(FC), gx₂ₖ₋₁)  # gx₂ₖ₋₁ = -gx₂ₖ₋₁ - conj(δₖ) * gx₂ₖ
        kaxpby!(n, one(FC), uₖ, -one(FC), gy₂ₖ₋₁)      # gy₂ₖ₋₁ = uₖ - gy₂ₖ₋₁
        kaxpy!(n, -conj(δₖ), gy₂ₖ, gy₂ₖ₋₁)             # gy₂ₖ₋₁ = gy₂ₖ₋₁ - conj(δₖ) * gy₂ₖ

        # gx₂ₖ₋₁ ↔ gx₂ₖ and gy₂ₖ₋₁ ↔ gy₂ₖ
        @kswap!(gx₂ₖ₋₁, gx₂ₖ)
        @kswap!(gy₂ₖ₋₁, gy₂ₖ)
      end

      # Update xₖ = Gxₖ * pₖ
      kaxpy!(m, π₂ₖ₋₁, gx₂ₖ₋₁, xₖ)
      kaxpy!(m, π₂ₖ  , gx₂ₖ  , xₖ)

      # Update yₖ = Gyₖ * pₖ
      kaxpy!(n, π₂ₖ₋₁, gy₂ₖ₋₁, yₖ)
      kaxpy!(n, π₂ₖ  , gy₂ₖ  , yₖ)

      # Compute vₖ₊₁ and uₖ₊₁
      MisI || mulorldiv!(vₖ₊₁, M, q, ldiv)  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
      NisI || mulorldiv!(uₖ₊₁, N, p, ldiv)  # γₖ₊₁uₖ₊₁ = NAᴴvₖ - βₖuₖ₋₁ - ᾱₖuₖ

      βₖ₊₁ = knorm_elliptic(m, vₖ₊₁, q)  # βₖ₊₁ = ‖vₖ₊₁‖_E
      γₖ₊₁ = knorm_elliptic(n, uₖ₊₁, p)  # γₖ₊₁ = ‖uₖ₊₁‖_F

      # βₖ₊₁ ≠ 0
      if βₖ₊₁ > btol
        kdiv!(m, q, βₖ₊₁)
        MisI || kdiv!(m, vₖ₊₁, βₖ₊₁)
      end
      # Note that if βₖ₊₁ == 0 then vₖ₊₁ = 0 and Auₖ ∈ Span{v₁, ..., vₖ}
      # We can keep vₖ₊₁ = 0 such that vₖ₊₁ ⊥ Span{v₁, ..., vₖ}

      # γₖ₊₁ ≠ 0
      if γₖ₊₁ > btol
        kdiv!(n, p, γₖ₊₁)
        NisI || kdiv!(n, uₖ₊₁, γₖ₊₁)
      end
      # Note that if γₖ₊₁ == 0 then uₖ₊₁ = 0 and Aᴴvₖ ∈ Span{u₁, ..., uₖ}
      # We can keep uₖ₊₁ = 0 such that uₖ₊₁ ⊥ Span{u₁, ..., uₖ}

      # Update M⁻¹vₖ and N⁻¹uₖ
      kcopy!(m, M⁻¹vₖ, q)  # M⁻¹vₖ ← q
      kcopy!(n, N⁻¹uₖ, p)  # N⁻¹uₖ ← p

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
      user_requested_exit = callback(workspace) :: Bool
      resid_decrease_lim = rNorm ≤ ε
      breakdown = βₖ₊₁ ≤ btol && γₖ₊₁ ≤ btol
      solved = resid_decrease_lim || resid_decrease_mach
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %7.1e  %.2fs\n", iter, rNorm, βₖ₊₁, γₖ₊₁, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = "inconsistent linear system")
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x and y
    warm_start && kaxpy!(m, one(FC), Δx, xₖ)
    warm_start && kaxpy!(n, one(FC), Δy, yₖ)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = !solved && breakdown
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
