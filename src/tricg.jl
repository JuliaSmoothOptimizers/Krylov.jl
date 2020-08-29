# An implementation of TriCG for the solution of symmetric and quasi-definite systems.
#
# This method is described in
#
# A. Montoison and D. Orban
# TriCG and TriMR: Two Iterative Methods for Symmetric Quasi-Definite Systems
# Cahier du GERAD G-2020-41, GERAD, Montreal, 2020. doi:10.13140/RG.2.2.12344.16645
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, April 2020.

export tricg

"""
TriCG solves the symmetric linear system

    [ τE    A ] [ x ] = [ b ]
    [  Aᵀ  νF ] [ y ]   [ c ],

where τ and ν are real numbers, E = M⁻¹ ≻ 0 and F = N⁻¹ ≻ 0.
TriCG could breakdown if `τ = 0` or `ν = 0`. It's recommended to use TriMR in these cases.

By default, TriCG solves symmetric and quasi-definite linear systems with τ = 1 and ν = -1.
If `flip = true`, TriCG solves another known variant of SQD systems where τ = -1 and ν = 1.
If `spd = true`, τ = ν = 1 and the associated symmetric and positive definite linear system is solved.
If `snd = true`, τ = ν = -1 and the associated symmetric and negative definite linear system is solved.
`τ` and `ν` are also keyword arguments that can be directly modified for more specific problems.

TriCG is based on the preconditioned orthogonal tridiagonalization process
and its relation with the preconditioned block-Lanczos process.

    [ M   O ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

TriCG stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖r₀‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if the `verbose` mode is enabled.
"""
function tricg(A, b :: AbstractVector{T}, c :: AbstractVector{T};
               M=opEye(), N=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T),
               spd :: Bool=false, snd :: Bool=false, flip :: Bool=false,
               τ :: T=one(T), ν :: T=-one(T), itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  verbose && @printf("TriCG: system of %d equations in %d variables\n", m+n, m+n)

  # Check flip, spd and snd parameters
  spd && flip && error("The matrix cannot be SPD and SQD")
  snd && flip && error("The matrix cannot be SND and SQD")
  spd && snd  && error("The matrix cannot be SPD and SND")

  # Check M == Iₘ and N == Iₙ
  MisI = isa(M, opEye)
  NisI = isa(N, opEye)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Determine the storage type of b
  S = typeof(b)

  # Initial solutions x₀ and y₀.
  xₖ = kzeros(S, m)
  yₖ = kzeros(S, n)

  iter = 0
  itmax == 0 && (itmax = m+n)
  
  # Initialize preconditioned orthogonal tridiagonalization process.
  M⁻¹vₖ₋₁ = kzeros(S, m)  # v₀ = 0
  N⁻¹uₖ₋₁ = kzeros(S, n)  # u₀ = 0

  # β₁Ev₁ = b ↔ β₁v₁ = Mb
  M⁻¹vₖ = copy(b)
  vₖ = M * M⁻¹vₖ
  βₖ = sqrt(@kdot(m, vₖ, M⁻¹vₖ))  # β₁ = ‖v₁‖_E
  if βₖ ≠ 0
    @kscal!(m, 1 / βₖ, M⁻¹vₖ)
    MisI || @kscal!(m, 1 / βₖ, vₖ)
  end

  # γ₁Fu₁ = c ↔ γ₁u₁ = Nb
  N⁻¹uₖ = copy(c)
  uₖ = N * N⁻¹uₖ
  γₖ = sqrt(@kdot(n, uₖ, N⁻¹uₖ))  # γ₁ = ‖u₁‖_F
  if γₖ ≠ 0
    @kscal!(n, 1 / γₖ, N⁻¹uₖ)
    NisI || @kscal!(n, 1 / γₖ, uₖ)
  end

  # Initialize directions Gₖ such that Lₖ(Gₖ)ᵀ = (Wₖ)ᵀ
  gx₂ₖ₋₁ = kzeros(S, m)
  gy₂ₖ₋₁ = kzeros(S, n)
  gx₂ₖ   = kzeros(S, m)
  gy₂ₖ   = kzeros(S, n)

  # Compute ‖r₀‖² = (γ₁)² + (β₁)²
  rNorm = sqrt(γₖ^2 + βₖ^2)
  rNorms = [rNorm;]
  ε = atol + rtol * rNorm

  verbose && @printf("%5s  %7s  %8s  %7s  %7s\n", "k", "‖rₖ‖", "αₖ", "βₖ₊₁", "γₖ₊₁")
  verbose && @printf("%5d  %7.1e  %8s  %7.1e  %7.1e\n", iter, rNorm, " ✗ ✗ ✗ ✗", βₖ, γₖ)

  # Set up workspace.
  d₂ₖ₋₃ = d₂ₖ₋₂ = zero(T)
  π₂ₖ₋₃ = π₂ₖ₋₂ = zero(T)
  δₖ₋₁ = zero(T)

  # Determine τ and ν associated to SQD, SPD or SND systems.
  flip && (τ = -one(T) ; ν =  one(T))
  spd  && (τ =  one(T) ; ν =  one(T))
  snd  && (τ = -one(T) ; ν = -one(T))

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)
    # Update iteration index.
    iter = iter + 1

    # Continue the orthogonal tridiagonalization process.
    # AUₖ  = EVₖTₖ    + βₖ₊₁Evₖ₊₁(eₖ)ᵀ = EVₖ₊₁Tₖ₊₁.ₖ
    # AᵀVₖ = FUₖ(Tₖ)ᵀ + γₖ₊₁Fuₖ₊₁(eₖ)ᵀ = FUₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    q = A  * uₖ  # Forms Evₖ₊₁ : q ← Auₖ
    p = Aᵀ * vₖ  # Forms Fuₖ₊₁ : p ← Aᵀvₖ

    if iter ≥ 2
      @kaxpy!(m, -γₖ, M⁻¹vₖ₋₁, q)  # q ← q - γₖ * M⁻¹vₖ₋₁
      @kaxpy!(n, -βₖ, N⁻¹uₖ₋₁, p)  # p ← p - βₖ * N⁻¹uₖ₋₁
    end

    αₖ = @kdot(m, vₖ, q)  # αₖ = qᵀvₖ

    @kaxpy!(m, -αₖ, M⁻¹vₖ, q)  # q ← q - αₖ * M⁻¹vₖ
    @kaxpy!(n, -αₖ, N⁻¹uₖ, p)  # p ← p - αₖ * N⁻¹uₖ

    # Update M⁻¹vₖ₋₁ and N⁻¹uₖ₋₁
    @. M⁻¹vₖ₋₁ = M⁻¹vₖ
    @. N⁻¹uₖ₋₁ = N⁻¹uₖ

    # Notations : Wₖ = [w₁ ••• wₖ] = [v₁ 0  ••• vₖ 0 ]
    #                                [0  u₁ ••• 0  uₖ]
    #
    # rₖ = [ b ] - [ τE    A ] [ xₖ ] = [ b ] - [ τE    A ] Wₖzₖ
    #      [ c ]   [  Aᵀ  νF ] [ yₖ ]   [ c ]   [  Aᵀ  νF ]
    #
    # block-Lanczos formulation : [ τE    A ] Wₖ = [ E   0 ] Wₖ₊₁Sₖ₊₁.ₖ
    #                             [  Aᵀ  νF ]      [ 0   F ]
    #
    # TriCG subproblem : (Wₖ)ᵀ * rₖ = 0 ↔ Sₖ.ₖzₖ = β₁e₁ + γ₁e₂
    #
    # Update the LDLᵀ factorization of Sₖ.ₖ.
    #
    # [ τ  α₁    γ₂ 0  •  •  •  •  0  ]
    # [ α₁ ν  β₂       •           •  ]
    # [    β₂ τ  α₂    γ₃ •        •  ]
    # [ γ₂    α₂ ν  β₃       •     •  ]
    # [ 0        β₃ •  •     •  •  •  ]
    # [ •  •  γ₃    •  •  •        0  ]
    # [ •     •        •  •  •     γₖ ]
    # [ •        •  •     •  •  βₖ    ]
    # [ •           •        βₖ τ  αₖ ]
    # [ 0  •  •  •  •  0  γₖ    αₖ ν  ]
    if iter == 1
      d₂ₖ₋₁ = τ
      δₖ    = αₖ / d₂ₖ₋₁
      d₂ₖ   = ν - δₖ^2 * d₂ₖ₋₁
    else
      σₖ    = βₖ / d₂ₖ₋₂
      ηₖ    = γₖ / d₂ₖ₋₃
      λₖ    = -(ηₖ * δₖ₋₁ * d₂ₖ₋₃) / d₂ₖ₋₂
      d₂ₖ₋₁ = τ - σₖ^2 * d₂ₖ₋₂
      δₖ    = (αₖ - λₖ * σₖ * d₂ₖ₋₂) / d₂ₖ₋₁
      d₂ₖ   = ν - ηₖ^2 * d₂ₖ₋₃ - λₖ^2 * d₂ₖ₋₂ - δₖ^2 * d₂ₖ₋₁
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

    # Solve Lₖ(Gₖ)ᵀ = (Wₖ)ᵀ.
    if iter == 1
      # [ 1  0 ] [ gx₁ gy₁ ] = [ v₁ 0  ]
      # [ δ₁ 1 ] [ gx₂ gy₂ ]   [ 0  u₁ ]
      @. gx₂ₖ₋₁ = vₖ
      @. gx₂ₖ   = - δₖ * gx₂ₖ₋₁
      @. gy₂ₖ   = uₖ
    else
      # [ 0  σₖ 1  0 ] [ gx₂ₖ₋₃ gy₂ₖ₋₃ ] = [ vₖ 0  ]
      # [ ηₖ λₖ δₖ 1 ] [ gx₂ₖ₋₂ gy₂ₖ₋₂ ]   [ 0  uₖ ]
      #                [ gx₂ₖ₋₁ gy₂ₖ₋₁ ]
      #                [ gx₂ₖ   gy₂ₖ   ]
      @. gx₂ₖ₋₁ = ηₖ * gx₂ₖ₋₁ + λₖ * gx₂ₖ
      @. gy₂ₖ₋₁ = ηₖ * gy₂ₖ₋₁ + λₖ * gy₂ₖ

      @. gx₂ₖ = vₖ - σₖ * gx₂ₖ
      @. gy₂ₖ =    - σₖ * gy₂ₖ

      @. gx₂ₖ₋₁ =    - gx₂ₖ₋₁ - δₖ * gx₂ₖ
      @. gy₂ₖ₋₁ = uₖ - gy₂ₖ₋₁ - δₖ * gy₂ₖ

      # g₂ₖ₋₃ == g₂ₖ and g₂ₖ₋₂ == g₂ₖ₋₁
      @kswap(gx₂ₖ₋₁, gx₂ₖ)
      @kswap(gy₂ₖ₋₁, gy₂ₖ)
    end

    # Update xₖ = Gxₖ * pₖ
    @. xₖ += π₂ₖ₋₁ * gx₂ₖ₋₁ + π₂ₖ * gx₂ₖ

    # Update yₖ = Gyₖ * pₖ
    @. yₖ += π₂ₖ₋₁ * gy₂ₖ₋₁ + π₂ₖ * gy₂ₖ

    # Compute vₖ₊₁ and uₖ₊₁
    vₖ₊₁ = M * q  # βₖ₊₁vₖ₊₁ = MAuₖ  - γₖvₖ₋₁ - αₖvₖ
    uₖ₊₁ = N * p  # γₖ₊₁uₖ₊₁ = NAᵀvₖ - βₖuₖ₋₁ - αₖuₖ

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

    # Update M⁻¹vₖ and N⁻¹uₖ
    @. M⁻¹vₖ = q
    @. N⁻¹uₖ = p

    # Compute ‖rₖ‖² = (γₖ₊₁ζ₂ₖ₋₁)² + (βₖ₊₁ζ₂ₖ)²
    rNorm = sqrt((γₖ₊₁ * (π₂ₖ₋₁ - δₖ*π₂ₖ))^2 + (βₖ₊₁ * π₂ₖ)^2)
    push!(rNorms, rNorm)

    # Update βₖ, γₖ, π₂ₖ₋₃, π₂ₖ₋₂, d₂ₖ₋₃, d₂ₖ₋₂, δₖ₋₁, vₖ, uₖ.
    βₖ    = βₖ₊₁
    γₖ    = γₖ₊₁
    π₂ₖ₋₃ = π₂ₖ₋₁
    π₂ₖ₋₂ = π₂ₖ
    d₂ₖ₋₃ = d₂ₖ₋₁
    d₂ₖ₋₂ = d₂ₖ
    δₖ₋₁  = δₖ
    MisI || (vₖ = vₖ₊₁)
    NisI || (uₖ = uₖ₊₁)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e  %8.1e  %7.1e  %7.1e\n", iter, rNorm, αₖ, βₖ₊₁, γₖ₊₁)
  end
  verbose && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (xₖ, yₖ, stats)
end
