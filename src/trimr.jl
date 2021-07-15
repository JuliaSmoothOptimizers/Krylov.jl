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
    (x, y, stats) = trimr(A, b::AbstractVector{T}, c::AbstractVector{T};
                          M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                          spd::Bool=false, snd::Bool=false, flip::Bool=false, sp::Bool=false,
                          τ::T=one(T), ν::T=-one(T), itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

TriMR solves the symmetric linear system

    [ τE    A ] [ x ] = [ b ]
    [  Aᵀ  νF ] [ y ]   [ c ],

where τ and ν are real numbers, E = M⁻¹ ≻ 0 and F = N⁻¹ ≻ 0.
TriMR handles saddle-point systems (`τ = 0` or `ν = 0`) and adjoint systems (`τ = 0` and `ν = 0`) without any risk of breakdown.

By default, TriMR solves symmetric and quasi-definite linear systems with τ = 1 and ν = -1.
If `flip = true`, TriMR solves another known variant of SQD systems where τ = -1 and ν = 1.
If `spd = true`, τ = ν = 1 and the associated symmetric and positive definite linear system is solved.
If `snd = true`, τ = ν = -1 and the associated symmetric and negative definite linear system is solved.
If `sp = true`, τ = 1, ν = 0 and the associated saddle-point linear system is solved.
`τ` and `ν` are also keyword arguments that can be directly modified for more specific problems.

TriMR is based on the preconditioned orthogonal tridiagonalization process
and its relation with the preconditioned block-Lanczos process.

    [ M   0 ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

TriMR stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

#### Reference

* A. Montoison and D. Orban, *TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems*, SIAM Journal on Scientific Computing, 43(4), pp. 2502--2525, 2021.
"""
function trimr(A, b :: AbstractVector{T}, c :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = TrimrSolver(A, b)
  trimr!(solver, A, b, c; kwargs...)
end

function trimr!(solver :: TrimrSolver{T,S}, A, b :: AbstractVector{T}, c :: AbstractVector{T};
                M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                spd :: Bool=false, snd :: Bool=false, flip :: Bool=false, sp :: Bool=false,
                τ :: T=one(T), ν :: T=-one(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("TriMR: system of %d equations in %d variables\n", m+n, m+n)

  # Check flip, sp, spd and snd parameters
  spd && flip && error("The matrix cannot be symmetric positive definite and symmetric quasi-definite !")
  spd && snd  && error("The matrix cannot be symmetric positive definite and symmetric negative definite !")
  spd && sp   && error("The matrix cannot be symmetric positive definite and a saddle-point !")
  snd && flip && error("The matrix cannot be symmetric negative definite and symmetric quasi-definite !")
  snd && sp   && error("The matrix cannot be symmetric negative definite and a saddle-point !")
  sp  && flip && error("The matrix cannot be symmetric quasi-definite and a saddle-point !")

  # Check M == Iₘ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  allocate_if(!MisI, solver, :vₖ, S, m)
  allocate_if(!NisI, solver, :uₖ, S, n)
  yₖ, N⁻¹uₖ₋₁, N⁻¹uₖ, p, xₖ, M⁻¹vₖ₋₁, M⁻¹vₖ, q = solver.yₖ, solver.N⁻¹uₖ₋₁, solver.N⁻¹uₖ, solver.p, solver.xₖ, solver.M⁻¹vₖ₋₁, solver.M⁻¹vₖ, solver.q
  gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ = solver.gy₂ₖ₋₃, solver.gy₂ₖ₋₂, solver.gy₂ₖ₋₁, solver.gy₂ₖ, solver.gx₂ₖ₋₃, solver.gx₂ₖ₋₂, solver.gx₂ₖ₋₁, solver.gx₂ₖ
  vₖ = MisI ? M⁻¹vₖ : solver.vₖ
  uₖ = NisI ? N⁻¹uₖ : solver.uₖ
  vₖ₊₁ = MisI ? q : M⁻¹vₖ₋₁
  uₖ₊₁ = NisI ? p : N⁻¹uₖ₋₁

  # Initial solutions x₀ and y₀.
  xₖ .= zero(T)
  yₖ .= zero(T)

  iter = 0
  itmax == 0 && (itmax = m+n)

  # Initialize preconditioned orthogonal tridiagonalization process.
  M⁻¹vₖ₋₁ .= zero(T)  # v₀ = 0
  N⁻¹uₖ₋₁ .= zero(T)  # u₀ = 0

  # β₁Ev₁ = b ↔ β₁v₁ = Mb
  M⁻¹vₖ .= b
  MisI || mul!(vₖ, M, M⁻¹vₖ)
  βₖ = sqrt(@kdot(m, vₖ, M⁻¹vₖ))  # β₁ = ‖v₁‖_E
  if βₖ ≠ 0
    @kscal!(m, 1 / βₖ, M⁻¹vₖ)
    MisI || @kscal!(m, 1 / βₖ, vₖ)
  end

  # γ₁Fu₁ = c ↔ γ₁u₁ = Nb
  N⁻¹uₖ .= c
  NisI || mul!(uₖ, N, N⁻¹uₖ)
  γₖ = sqrt(@kdot(n, uₖ, N⁻¹uₖ))  # γ₁ = ‖u₁‖_F
  if γₖ ≠ 0
    @kscal!(n, 1 / γₖ, N⁻¹uₖ)
    NisI || @kscal!(n, 1 / γₖ, uₖ)
  end

  # Initialize directions Gₖ such that (GₖRₖ)ᵀ = (Wₖ)ᵀ.
  gx₂ₖ₋₃ .= zero(T)
  gy₂ₖ₋₃ .= zero(T)
  gx₂ₖ₋₂ .= zero(T)
  gy₂ₖ₋₂ .= zero(T)
  gx₂ₖ₋₁ .= zero(T)
  gy₂ₖ₋₁ .= zero(T)
  gx₂ₖ   .= zero(T)
  gy₂ₖ   .= zero(T)

  # Compute ‖r₀‖² = (γ₁)² + (β₁)²
  rNorm = sqrt(γₖ^2 + βₖ^2)
  rNorms = history ? [rNorm] : T[]
  ε = atol + rtol * rNorm

  (verbose > 0) && @printf("%5s  %7s  %8s  %7s  %7s\n", "k", "‖rₖ‖", "αₖ", "βₖ₊₁", "γₖ₊₁")
  display(iter, verbose) && @printf("%5d  %7.1e  %8s  %7.1e  %7.1e\n", iter, rNorm, " ✗ ✗ ✗ ✗", βₖ, γₖ)

  # Set up workspace.
  old_s₁ₖ = old_s₂ₖ = old_s₃ₖ = old_s₄ₖ = zero(T)
  old_c₁ₖ = old_c₂ₖ = old_c₃ₖ = old_c₄ₖ = zero(T)
  σbar₂ₖ₋₂ = ηbar₂ₖ₋₃ = λbar₂ₖ₋₃ = μ₂ₖ₋₅ = λ₂ₖ₋₄ = μ₂ₖ₋₄ = zero(T)
  πbar₂ₖ₋₁ = βₖ
  πbar₂ₖ = γₖ

  # Determine τ and ν associated to SQD, SPD or SND systems.
  flip && (τ = -one(T) ; ν =  one(T))
  spd  && (τ =  one(T) ; ν =  one(T))
  snd  && (τ = -one(T) ; ν = -one(T))
  sp   && (τ =  one(T) ; ν = zero(T))

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  θbarₖ = δbar₂ₖ₋₁ = δbar₂ₖ = σbar₂ₖ₋₁ = σbar₂ₖ = λbar₂ₖ₋₁ = ηbar₂ₖ₋₁ = zero(T)

  while !(solved || tired)
    # Update iteration index.
    iter = iter + 1

    # Continue the orthogonal tridiagonalization process.
    # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
    # AᵀVₖ = FUₖ(Tₖ)ᵀ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    mul!(q, A , uₖ)  # Forms Evₖ₊₁ : q ← Auₖ
    mul!(p, Aᵀ, vₖ)  # Forms Fuₖ₊₁ : p ← Aᵀvₖ

    if iter ≥ 2
      @kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q)  # q ← q - γₖ * M⁻¹vₖ₋₁
      @kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p)  # p ← p - βₖ * N⁻¹uₖ₋₁
    end

    αₖ = @kdot(m, vₖ, q)  # αₖ = qᵀvₖ

    @kaxpy!(m, -αₖ, M⁻¹vₖ, q)  # q ← q - αₖ * M⁻¹vₖ
    @kaxpy!(n, -αₖ, N⁻¹uₖ, p)  # p ← p - αₖ * N⁻¹uₖ

    # Compute vₖ₊₁ and uₖ₊₁
    MisI || mul!(vₖ₊₁, M, q)  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
    NisI || mul!(uₖ₊₁, N, p)  # γₖ₊₁uₖ₊₁ = NAᵀvₖ - βₖuₖ₋₁ - αₖuₖ

    βₖ₊₁ = sqrt(@kdot(m, vₖ₊₁, q))  # βₖ₊₁ = ‖vₖ₊₁‖_E
    γₖ₊₁ = sqrt(@kdot(n, uₖ₊₁, p))  # γₖ₊₁ = ‖uₖ₊₁‖_F

    if βₖ₊₁ ≠ 0
      @kscal!(m, one(T) / βₖ₊₁, q)
      MisI || @kscal!(m, one(T) / βₖ₊₁, vₖ₊₁)
    end

    if γₖ₊₁ ≠ 0
      @kscal!(n, one(T) / γₖ₊₁, p)
      NisI || @kscal!(n, one(T) / γₖ₊₁, uₖ₊₁)
    end

    # Notations : Wₖ = [w₁ ••• wₖ] = [v₁ 0  ••• vₖ 0 ]
    #                                [0  u₁ ••• 0  uₖ]
    #
    # rₖ = [ b ] - [ τE    A ] [ xₖ ] = [ b ] - [ τE    A ] Wₖzₖ
    #      [ c ]   [  Aᵀ  νF ] [ yₖ ]   [ c ]   [  Aᵀ  νF ]
    #
    # block-Lanczos formulation : [ τE    A ] Wₖ = [ E   0 ] Wₖ₊₁Sₖ₊₁.ₖ
    #                             [  Aᵀ  νF ]      [ 0   F ]
    #
    # TriMR subproblem : min ‖ rₖ ‖ ↔ min ‖ Sₖ₊₁.ₖzₖ - β₁e₁ + γ₁e₂ ‖
    #
    # Update the QR factorization of Sₖ₊₁.ₖ = Qₖ [ Rₖ ].
    #                                            [ Oᵀ ]
    if iter == 1
      θbarₖ    = αₖ
      δbar₂ₖ₋₁ = τ
      δbar₂ₖ   = ν
      σbar₂ₖ₋₁ = αₖ
      σbar₂ₖ   = βₖ₊₁
      λbar₂ₖ₋₁ = γₖ₊₁
      ηbar₂ₖ₋₁ = zero(T)
    else
      # Apply previous reflections
      #        [ 1                    ][ 1                    ][ c₂.ₖ₋₁  s₂.ₖ₋₁       ][ 1                    ]
      # Ζₖ₋₁ = [    c₄.ₖ₋₁  s₄.ₖ₋₁    ][    c₃.ₖ₋₁     s₃.ₖ₋₁ ][ s₂.ₖ₋₁ -c₂.ₖ₋₁       ][    c₁.ₖ₋₁     s₁.ₖ₋₁ ]
      #        [    s₄.ₖ₋₁ -c₄.ₖ₋₁    ][            1         ][                 1    ][            1         ]
      #        [                    1 ][    s₃.ₖ₋₁    -c₃.ₖ₋₁ ][                    1 ][    s₁.ₖ₋₁    -c₁.ₖ₋₁ ]
      #
      #        [ δbar₂ₖ₋₃  σbar₂ₖ₋₃ ηbar₂ₖ₋₃ λbar₂ₖ₋₃  0      0  ]   [ δ₂ₖ₋₃   σ₂ₖ₋₃  η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃       0      ]
      # Ζₖ₋₁ * [ θbarₖ₋₁   δbar₂ₖ₋₂ σbar₂ₖ₋₂    0      0      0  ] = [  0      δ₂ₖ₋₂  σ₂ₖ₋₂     η₂ₖ₋₂     λ₂ₖ₋₂     μ₂ₖ₋₂    ]
      #        [    0         βₖ       τ        αₖ     0    γₖ₊₁ ]   [  0        0    δbar₂ₖ₋₁  σbar₂ₖ₋₁  ηbar₂ₖ₋₁  λbar₂ₖ₋₁ ]
      #        [    γₖ        0        αₖ       ν     βₖ₊₁    0  ]   [  0        0    θbarₖ     δbar₂ₖ    σbar₂ₖ      0      ]
      #
      # [ 1                    ] [ ηbar₂ₖ₋₃ λbar₂ₖ₋₃  0      0  ]   [ ηbar₂ₖ₋₃  λbar₂ₖ₋₃    0        0   ]
      # [    c₁.ₖ₋₁     s₁.ₖ₋₁ ] [ σbar₂ₖ₋₂    0      0      0  ] = [ σbis₂ₖ₋₂  ηbis₂ₖ₋₂  λbis₂ₖ₋₂   0   ]
      # [            1         ] [    τ        αₖ     0    γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
      # [    s₁.ₖ₋₁    -c₁.ₖ₋₁ ] [    αₖ       ν     βₖ₊₁    0  ]   [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]
      σbis₂ₖ₋₂ = old_c₁ₖ * σbar₂ₖ₋₂ + old_s₁ₖ * αₖ
      ηbis₂ₖ₋₂ =                      old_s₁ₖ * ν
      λbis₂ₖ₋₂ =                      old_s₁ₖ * βₖ₊₁
      θbisₖ    = old_s₁ₖ * σbar₂ₖ₋₂ - old_c₁ₖ * αₖ
      δbis₂ₖ   =                    - old_c₁ₖ * ν
      σbis₂ₖ   =                    - old_c₁ₖ * βₖ₊₁
      # [ c₂.ₖ₋₁  s₂.ₖ₋₁       ] [ ηbar₂ₖ₋₃  λbar₂ₖ₋₃    0        0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]
      # [ s₂.ₖ₋₁ -c₂.ₖ₋₁       ] [ σbis₂ₖ₋₂  ηbis₂ₖ₋₂  λbis₂ₖ₋₂   0   ] = [ σhat₂ₖ₋₂  ηhat₂ₖ₋₂  λhat₂ₖ₋₂   0   ]
      # [                 1    ] [   τ        αₖ         0       γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
      # [                    1 ] [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]   [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]
      η₂ₖ₋₃    = old_c₂ₖ * ηbar₂ₖ₋₃ + old_s₂ₖ * σbis₂ₖ₋₂
      λ₂ₖ₋₃    = old_c₂ₖ * λbar₂ₖ₋₃ + old_s₂ₖ * ηbis₂ₖ₋₂
      μ₂ₖ₋₃    =                      old_s₂ₖ * λbis₂ₖ₋₂
      σhat₂ₖ₋₂ = old_s₂ₖ * ηbar₂ₖ₋₃ - old_c₂ₖ * σbis₂ₖ₋₂
      ηhat₂ₖ₋₂ = old_s₂ₖ * λbar₂ₖ₋₃ - old_c₂ₖ * ηbis₂ₖ₋₂
      λhat₂ₖ₋₂ =                    - old_c₂ₖ * λbis₂ₖ₋₂
      # [ 1                    ] [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]
      # [    c₃.ₖ₋₁     s₃.ₖ₋₁ ] [ σhat₂ₖ₋₂  ηhat₂ₖ₋₂  λhat₂ₖ₋₂   0   ] = [ σtmp₂ₖ₋₂  ηtmp₂ₖ₋₂  λtmp₂ₖ₋₂   0   ]
      # [            1         ] [   τ        αₖ         0       γₖ₊₁ ]   [   τ        αₖ         0       γₖ₊₁ ]
      # [    s₃.ₖ₋₁    -c₃.ₖ₋₁ ] [  θbisₖ    δbis₂ₖ    σbis₂ₖ     0   ]   [  θbarₖ    δbar₂ₖ    σbar₂ₖ     0   ]
      σtmp₂ₖ₋₂ = old_c₃ₖ * σhat₂ₖ₋₂ + old_s₃ₖ * θbisₖ
      ηtmp₂ₖ₋₂ = old_c₃ₖ * ηhat₂ₖ₋₂ + old_s₃ₖ * δbis₂ₖ
      λtmp₂ₖ₋₂ = old_c₃ₖ * λhat₂ₖ₋₂ + old_s₃ₖ * σbis₂ₖ
      θbarₖ    = old_s₃ₖ * σhat₂ₖ₋₂ - old_c₃ₖ * θbisₖ
      δbar₂ₖ   = old_s₃ₖ * ηhat₂ₖ₋₂ - old_c₃ₖ * δbis₂ₖ
      σbar₂ₖ   = old_s₃ₖ * λhat₂ₖ₋₂ - old_c₃ₖ * σbis₂ₖ
      # [ 1                    ] [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃      0   ]   [ η₂ₖ₋₃     λ₂ₖ₋₃     μ₂ₖ₋₃       0      ]
      # [    c₄.ₖ₋₁  s₄.ₖ₋₁    ] [ σtmp₂ₖ₋₂  ηtmp₂ₖ₋₂  λtmp₂ₖ₋₂   0   ] = [ σ₂ₖ₋₂     η₂ₖ₋₂     λ₂ₖ₋₂     μ₂ₖ₋₂    ]
      # [    s₄.ₖ₋₁ -c₄.ₖ₋₁    ] [   τ        αₖ         0       γₖ₊₁ ]   [ δbar₂ₖ₋₁  σbar₂ₖ₋₁  ηbar₂ₖ₋₁  λbar₂ₖ₋₁ ]
      # [                    1 ] [  θbarₖ    δbar₂ₖ    σbar₂ₖ     0   ]   [ θbarₖ     δbar₂ₖ    σbar₂ₖ      0      ]
      σ₂ₖ₋₂    = old_c₄ₖ * σtmp₂ₖ₋₂ + old_s₄ₖ * τ
      η₂ₖ₋₂    = old_c₄ₖ * ηtmp₂ₖ₋₂ + old_s₄ₖ * αₖ
      λ₂ₖ₋₂    = old_c₄ₖ * λtmp₂ₖ₋₂
      μ₂ₖ₋₂    =                      old_s₄ₖ * γₖ₊₁
      δbar₂ₖ₋₁ = old_s₄ₖ * σtmp₂ₖ₋₂ - old_c₄ₖ * τ
      σbar₂ₖ₋₁ = old_s₄ₖ * ηtmp₂ₖ₋₂ - old_c₄ₖ * αₖ
      ηbar₂ₖ₋₁ = old_s₄ₖ * λtmp₂ₖ₋₂
      λbar₂ₖ₋₁ =                    - old_c₄ₖ * γₖ₊₁
    end

    # [ 1                ] [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]   [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]
    # [    c₁.ₖ     s₁.ₖ ] [  θbarₖ     δbar₂ₖ  ] = [   θₖ       δbar₂ₖ  ]
    # [          1       ] [   0         βₖ₊₁   ]   [   0         βₖ₊₁   ]
    # [    s₁.ₖ    -c₁.ₖ ] [  γₖ₊₁        0     ]   [   0         gₖ     ]
    (c₁ₖ, s₁ₖ, θₖ) = sym_givens(θbarₖ, γₖ₊₁)
    gₖ     = s₁ₖ * δbar₂ₖ
    δbar₂ₖ = c₁ₖ * δbar₂ₖ

    # [ c₂.ₖ  s₂.ₖ       ] [ δbar₂ₖ₋₁  σbar₂ₖ₋₁ ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]
    # [ s₂.ₖ -c₂.ₖ       ] [   θₖ       δbar₂ₖ  ] = [  0     δbis₂ₖ ]
    # [             1    ] [   0         βₖ₊₁   ]   [  0      βₖ₊₁  ]
    # [                1 ] [   0         gₖ     ]   [  0       gₖ   ]
    (c₂ₖ, s₂ₖ, δ₂ₖ₋₁) = sym_givens(δbar₂ₖ₋₁, θₖ)
    σ₂ₖ₋₁  = c₂ₖ * σbar₂ₖ₋₁ + s₂ₖ * δbar₂ₖ
    δbis₂ₖ = s₂ₖ * σbar₂ₖ₋₁ - c₂ₖ * δbar₂ₖ

    # [ 1                ] [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]
    # [    c₃.ₖ     s₃.ₖ ] [  0     δbis₂ₖ ] = [  0     δhat₂ₖ ]
    # [          1       ] [  0      βₖ₊₁  ]   [  0      βₖ₊₁  ]
    # [    s₃.ₖ    -c₃.ₖ ] [  0       gₖ   ]   [  0       0    ]
    (c₃ₖ, s₃ₖ, δhat₂ₖ) = sym_givens(δbis₂ₖ, gₖ)

    # [ 1                ] [ δ₂ₖ₋₁  σ₂ₖ₋₁  ]   [ δ₂ₖ₋₁  σ₂ₖ₋₁ ]
    # [    c₄.ₖ  s₄.ₖ    ] [  0     δhat₂ₖ ] = [  0      δ₂ₖ  ]
    # [    s₄.ₖ -c₄.ₖ    ] [  0      βₖ₊₁  ]   [  0       0   ]
    # [                1 ] [  0       0    ]   [  0       0   ]
    (c₄ₖ, s₄ₖ, δ₂ₖ) = sym_givens(δhat₂ₖ, βₖ₊₁)

    # Solve Gₖ = Wₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Gₖ)ᵀ = (Wₖ)ᵀ.
    if iter == 1
      # [ δ₁  0  ] [ gx₁ gy₁ ] = [ v₁ 0  ]
      # [ σ₁  δ₂ ] [ gx₂ gy₂ ]   [ 0  u₁ ]
      @. gx₂ₖ₋₁ = vₖ / δ₂ₖ₋₁
      @. gx₂ₖ   = - σ₂ₖ₋₁ / δ₂ₖ * gx₂ₖ₋₁
      @. gy₂ₖ   = uₖ / δ₂ₖ
    elseif iter == 2
      # [ η₁ σ₂ δ₃ 0  ] [ gx₁ gy₁ ] = [ v₂ 0  ]
      # [ λ₁ η₂ σ₃ δ₄ ] [ gx₂ gy₂ ]   [ 0  u₂ ]
      #                 [ gx₃ gy₃ ]
      #                 [ gx₄ gy₄ ]
      @kswap(gx₂ₖ₋₃, gx₂ₖ₋₁)
      @kswap(gx₂ₖ₋₂, gx₂ₖ)
      @kswap(gy₂ₖ₋₂, gy₂ₖ)
      @. gx₂ₖ₋₁ = (vₖ - η₂ₖ₋₃ * gx₂ₖ₋₃ - σ₂ₖ₋₂ * gx₂ₖ₋₂                 ) / δ₂ₖ₋₁
      @. gx₂ₖ   = (   - λ₂ₖ₋₃ * gx₂ₖ₋₃ - η₂ₖ₋₂ * gx₂ₖ₋₂ - σ₂ₖ₋₁ * gx₂ₖ₋₁) / δ₂ₖ
      @. gy₂ₖ₋₁ = (   - η₂ₖ₋₃ * gy₂ₖ₋₃ - σ₂ₖ₋₂ * gy₂ₖ₋₂                 ) / δ₂ₖ₋₁
      @. gy₂ₖ   = (uₖ - λ₂ₖ₋₃ * gy₂ₖ₋₃ - η₂ₖ₋₂ * gy₂ₖ₋₂ - σ₂ₖ₋₁ * gy₂ₖ₋₁) / δ₂ₖ
    else
      # μ₂ₖ₋₅ * gx₂ₖ₋₅ + λ₂ₖ₋₄ * gx₂ₖ₋₄ + η₂ₖ₋₃ * gx₂ₖ₋₃ + σ₂ₖ₋₂ * gx₂ₖ₋₂ + δ₂ₖ₋₁ * gx₂ₖ₋₁              = vₖ
      #                  μ₂ₖ₋₄ * gx₂ₖ₋₄ + λ₂ₖ₋₃ * gx₂ₖ₋₃ + η₂ₖ₋₂ * gx₂ₖ₋₂ + σ₂ₖ₋₁ * gx₂ₖ₋₁ + δ₂ₖ * gx₂ₖ = 0
      g₂ₖ₋₁ = g₂ₖ₋₅ = gx₂ₖ₋₃; g₂ₖ = g₂ₖ₋₄ = gx₂ₖ₋₂; g₂ₖ₋₃ = gx₂ₖ₋₁; g₂ₖ₋₂ = gx₂ₖ
      @. g₂ₖ₋₁ = (vₖ - μ₂ₖ₋₅ * g₂ₖ₋₅ - λ₂ₖ₋₄ * g₂ₖ₋₄ - η₂ₖ₋₃ * g₂ₖ₋₃ - σ₂ₖ₋₂ * g₂ₖ₋₂                ) / δ₂ₖ₋₁
      @. g₂ₖ   = (                   - μ₂ₖ₋₄ * g₂ₖ₋₄ - λ₂ₖ₋₃ * g₂ₖ₋₃ - η₂ₖ₋₂ * g₂ₖ₋₂ - σ₂ₖ₋₁ * g₂ₖ₋₁) / δ₂ₖ
      @kswap(gx₂ₖ₋₃, gx₂ₖ₋₁)
      @kswap(gx₂ₖ₋₂, gx₂ₖ)
      # μ₂ₖ₋₅ * gy₂ₖ₋₅ + λ₂ₖ₋₄ * gy₂ₖ₋₄ + η₂ₖ₋₃ * gy₂ₖ₋₃ + σ₂ₖ₋₂ * gy₂ₖ₋₂ + δ₂ₖ₋₁ * gy₂ₖ₋₁              = 0
      #                  μ₂ₖ₋₄ * gy₂ₖ₋₄ + λ₂ₖ₋₃ * gy₂ₖ₋₃ + η₂ₖ₋₂ * gy₂ₖ₋₂ + σ₂ₖ₋₁ * gy₂ₖ₋₁ + δ₂ₖ * gy₂ₖ = uₖ
      g₂ₖ₋₁ = g₂ₖ₋₅ = gy₂ₖ₋₃; g₂ₖ = g₂ₖ₋₄ = gy₂ₖ₋₂; g₂ₖ₋₃ = gy₂ₖ₋₁; g₂ₖ₋₂ = gy₂ₖ
      @. g₂ₖ₋₁ = (     - μ₂ₖ₋₅ * g₂ₖ₋₅ - λ₂ₖ₋₄ * g₂ₖ₋₄ - η₂ₖ₋₃ * g₂ₖ₋₃ - σ₂ₖ₋₂ * g₂ₖ₋₂                ) / δ₂ₖ₋₁
      @. g₂ₖ   = (uₖ                   - μ₂ₖ₋₄ * g₂ₖ₋₄ - λ₂ₖ₋₃ * g₂ₖ₋₃ - η₂ₖ₋₂ * g₂ₖ₋₂ - σ₂ₖ₋₁ * g₂ₖ₋₁) / δ₂ₖ
      @kswap(gy₂ₖ₋₃, gy₂ₖ₋₁)
      @kswap(gy₂ₖ₋₂, gy₂ₖ)
    end

    # Update p̅ₖ = (Qₖ)ᵀ * (β₁e₁ + γ₁e₂)
    πbis₂ₖ   = c₁ₖ * πbar₂ₖ
    πbis₂ₖ₊₂ = s₁ₖ * πbar₂ₖ
    #
    π₂ₖ₋₁  = c₂ₖ * πbar₂ₖ₋₁ + s₂ₖ * πbis₂ₖ
    πhat₂ₖ = s₂ₖ * πbar₂ₖ₋₁ - c₂ₖ * πbis₂ₖ
    #
    πtmp₂ₖ   = c₃ₖ * πhat₂ₖ + s₃ₖ * πbis₂ₖ₊₂
    πbar₂ₖ₊₂ = s₃ₖ * πhat₂ₖ - c₃ₖ * πbis₂ₖ₊₂
    #
    π₂ₖ      = c₄ₖ * πtmp₂ₖ
    πbar₂ₖ₊₁ = s₄ₖ * πtmp₂ₖ

    # Update xₖ = Gxₖ * pₖ
    @. xₖ += π₂ₖ₋₁ * gx₂ₖ₋₁ + π₂ₖ * gx₂ₖ

    # Update yₖ = Gyₖ * pₖ
    @. yₖ += π₂ₖ₋₁ * gy₂ₖ₋₁ + π₂ₖ * gy₂ₖ

    # Compute ‖rₖ‖² = (πbar₂ₖ₊₁)² + (πbar₂ₖ₊₂)²
    rNorm = sqrt(πbar₂ₖ₊₁^2 + πbar₂ₖ₊₂^2)
    history && push!(rNorms, rNorm)

    # Update vₖ and uₖ
    MisI || (vₖ .= vₖ₊₁)
    NisI || (uₖ .= uₖ₊₁)

    # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
    @. M⁻¹vₖ₋₁ = M⁻¹vₖ
    @. N⁻¹uₖ₋₁ = N⁻¹uₖ

    # Update M⁻¹vₖ and N⁻¹uₖ
    @. M⁻¹vₖ = q
    @. N⁻¹uₖ = p

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

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    display(iter, verbose) && @printf("%5d  %7.1e  %8.1e  %7.1e  %7.1e\n", iter, rNorm, αₖ, βₖ₊₁, γₖ₊₁)
  end
  (verbose > 0) && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (xₖ, yₖ, stats)
end
