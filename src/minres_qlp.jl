# An implementation of MINRES-QLP.
#
# This method is described in
#
# S.-C. T. Choi, Iterative methods for singular linear equations and least-squares problems.
# PHD thesis, ICME, Stanford University, 2006.
#
# S.-C. T. Choi, C. C. Paige and M. A. Saunders, MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems.
# SIAM Journal on Scientific Computing, Vol. 33(4), pp. 1810--1836, 2011.
#
# S.-C. T. Choi and M. A. Saunders, MINRES-QLP for symmetric and Hermitian linear equations and least-squares problems.
# ACM Transactions on Mathematical Software, 2014.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2019.

export minres_qlp

"""
MINRES-QLP is the only method based on the Lanczos process that returns the minimum-norm
solution on singular inconsistent systems (A + λI)x = b, where λ is a shift parameter.
It is significantly more complex but can be more reliable than MINRES when A is ill-conditioned.
"""
function minres_qlp(A, b :: AbstractVector{T};
                    atol :: T=√eps(T), rtol :: T=√eps(T), λ ::T=zero(T),
                    itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("MINRES-QLP: system of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial state.
  x = kzeros(S, n)
  rNorm = @knrm2(n, b)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  ArNorms = T[]
  κ = zero(T)
  verbose && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖Arₖ₋₁‖")
  verbose && @printf("%5d  %7.1e  %7s\n", iter, rNorm, "✗ ✗ ✗ ✗")

  # Set up workspace.
  vₖ₋₁  = kzeros(S, n)
  βₖ    = rNorm
  vₖ    = b / βₖ
  ζbarₖ = βₖ
  ξₖ₋₁  = zero(T)
  τₖ₋₂ = τₖ₋₁ = τₖ = zero(T)
  ψbarₖ₋₂ = zero(T)
  μbisₖ₋₂ = μbarₖ₋₁ = zero(T)
  wₖ₋₁  = kzeros(S, n)
  wₖ    = kzeros(S, n)
  cₖ₋₂  = cₖ₋₁ = cₖ = zero(T)  # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
  sₖ₋₂  = sₖ₋₁ = sₖ = zero(T)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ

  # Stopping criterion.
  solved = rNorm ≤ ε
  inconsistent = false
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired || inconsistent)
    # Update iteration index.
    iter = iter + 1

    # Continue the Lanczos process.
    # (A - λI)Vₖ = Vₖ₊₁Tₖ₊₁.ₖ
    # βₖ₊₁vₖ₊₁ = (A - λI)vₖ - αₖvₖ - βₖvₖ₋₁

    p = A * vₖ                  # p ← Avₖ
    if λ ≠ 0
      @kaxpy!(n, -λ, vₖ, p)     # p ← p - λvₖ
    end
    if iter ≥ 2
      @kaxpy!(n, -βₖ, vₖ₋₁, p)  # p ← p - βₖvₖ₋₁
    end
    αₖ = @kdot(n, vₖ, p)        # αₖ = pᵀvₖ
    @kaxpy!(n, -αₖ, vₖ, p)      # p ← p - αₖvₖ
    βₖ₊₁ = @knrm2(n, p)         # βₖ₊₁ = ‖ p ‖

    # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
    #                                            [ Oᵀ ]
    #
    # [ α₁ β₂ 0  •  •  •   0  ]      [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]
    # [ β₂ α₂ β₃ •         •  ]      [ 0  λ₂ γ₂ •  •     •  ]
    # [ 0  •  •  •  •      •  ]      [ •  •  λ₃ •  •  •  •  ]
    # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
    # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
    # [ •        •  •  •   βₖ ]      [ •           •  • γₖ₋₁]
    # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  λₖ ]
    # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
    #
    # If k = 1, we don't have any previous reflexion.
    # If k = 2, we apply the last reflexion.
    # If k ≥ 3, we only apply the two previous reflexions.

    # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
    if iter ≥ 3
      # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
      # [sₖ₋₂ -cₖ₋₂] [βₖ]   [γbarₖ₋₁]
      ϵₖ₋₂    =  sₖ₋₂ * βₖ
      γbarₖ₋₁ = -cₖ₋₂ * βₖ
    end
    # Apply previous Givens reflections Qₖ₋₁.ₖ
    if iter ≥ 2
      iter == 2 && (γbarₖ₋₁ = βₖ)
      # [cₖ₋₁  sₖ₋₁] [γbarₖ₋₁] = [γₖ₋₁ ]
      # [sₖ₋₁ -cₖ₋₁] [   αₖ  ]   [λbarₖ]
      γₖ₋₁  = cₖ₋₁ * γbarₖ₋₁ + sₖ₋₁ * αₖ
      λbarₖ = sₖ₋₁ * γbarₖ₋₁ - cₖ₋₁ * αₖ
    end
    iter == 1 && (λbarₖ = αₖ)

    # Compute and apply current Givens reflection Qₖ.ₖ₊₁
    # [cₖ  sₖ] [λbarₖ] = [λₖ]
    # [sₖ -cₖ] [βₖ₊₁ ]   [0 ]
    (cₖ, sₖ, λₖ) = sym_givens(λbarₖ, βₖ₊₁)

    # Compute [   zₖ  ] = (Qₖ)ᵀβ₁e₁
    #         [ζbarₖ₊₁]
    #
    # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
    # [sₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
    ζₖ      = cₖ * ζbarₖ
    ζbarₖ₊₁ = sₖ * ζbarₖ

    # Update the LQ factorization of Rₖ = LₖPₖ.
    # [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]   [ μ₁   0    •    •     •      •      0  ]
    # [ 0  λ₂ γ₂ •  •     •  ]   [ ψ₁   μ₂   •                        •  ]
    # [ •  •  λ₃ •  •  •  •  ]   [ ρ₁   ψ₂   μ₃   •                   •  ]
    # [ •     •  •  •  •  0  ] = [ 0    •    •    •     •             •  ] Pₖ
    # [ •        •  •  • ϵₖ₋₂]   [ •    •    •    •   μₖ₋₂     •      •  ]
    # [ •           •  • γₖ₋₁]   [ •         •    •   ψₖ₋₂  μbisₖ₋₁   0  ]
    # [ 0  •  •  •  •  0  λₖ ]   [ 0    •    •    0   ρₖ₋₂  ψbarₖ₋₁ μbarₖ]

    if iter == 1
      μbarₖ = λₖ
    elseif iter == 2
      # [μbar₁ γ₁] [cp₂  sp₂] = [μbis₁   0  ]
      # [  0   λ₂] [sp₂ -cp₂]   [ψbar₁ μbar₂]
      (cpₖ, spₖ, μbisₖ₋₁) = sym_givens(μbarₖ₋₁, γₖ₋₁)
      ψbarₖ₋₁ =  spₖ * λₖ
      μbarₖ   = -cpₖ * λₖ
    else
      # [μbisₖ₋₂   0     ϵₖ₋₂] [cpₖ  0   spₖ]   [μₖ₋₂   0     0 ]
      # [ψbarₖ₋₂ μbarₖ₋₁ γₖ₋₁] [ 0   1    0 ] = [ψₖ₋₂ μbarₖ₋₁ θₖ]
      # [  0       0      λₖ ] [spₖ  0  -cpₖ]   [ρₖ₋₂   0     ηₖ]
      (cpₖ, spₖ, μₖ₋₂) = sym_givens(μbisₖ₋₂, ϵₖ₋₂)
      ψₖ₋₂ =  cpₖ * ψbarₖ₋₂ + spₖ * γₖ₋₁
      θₖ   =  spₖ * ψbarₖ₋₂ - cpₖ * γₖ₋₁
      ρₖ₋₂ =  spₖ * λₖ
      ηₖ   = -cpₖ * λₖ

      # [μₖ₋₂   0     0 ] [1   0    0 ]   [μₖ₋₂   0       0  ]
      # [ψₖ₋₂ μbarₖ₋₁ θₖ] [0  cdₖ  sdₖ] = [ψₖ₋₂ μbisₖ₋₁   0  ]
      # [ρₖ₋₂   0     ηₖ] [0  sdₖ -cdₖ]   [ρₖ₋₂ ψbarₖ₋₁ μbarₖ]
      (cdₖ, sdₖ, μbisₖ₋₁) = sym_givens(μbarₖ₋₁, θₖ)
      ψbarₖ₋₁ =  sdₖ * ηₖ
      μbarₖ   = -cdₖ * ηₖ
    end

    # Compute Lₖtₖ = zₖ
    # [ μ₁   0    •    •     •      •      0  ] [τ₁]   [ζ₁]
    # [ ψ₁   μ₂   •                        •  ] [τ₂]   [ζ₂]
    # [ ρ₁   ψ₂   μ₃   •                   •  ] [τ₃]   [ζ₃]
    # [ 0    •    •    •     •             •  ] [••] = [••]
    # [ •    •    •    •   μₖ₋₂     •      •  ] [••]   [••]
    # [ •         •    •   ψₖ₋₂  μbisₖ₋₁   0  ] [••]   [••]
    # [ 0    •    •    0   ρₖ₋₂  ψbarₖ₋₁ μbarₖ] [τₖ]   [ζₖ]
    if iter == 1
      τₖ = ζₖ / μbarₖ
    elseif iter == 2
      τₖ₋₁ = τₖ
      τₖ₋₁ = τₖ₋₁ * μbarₖ₋₁ / μbisₖ₋₁
      ξₖ   = ζₖ
      τₖ   = (ξₖ - ψbarₖ₋₁ * τₖ₋₁) / μbarₖ
    else
      τₖ₋₂ = τₖ₋₁
      τₖ₋₂ = τₖ₋₂ * μbisₖ₋₂ / μₖ₋₂
      τₖ₋₁ = τₖ
      τₖ₋₁ = (ξₖ₋₁ - ψₖ₋₂ * τₖ₋₂) / μbisₖ₋₁
      ξₖ   = ζₖ - ρₖ₋₂ * τₖ₋₂
      τₖ   = (ξₖ - ψbarₖ₋₁ * τₖ₋₁) / μbarₖ
    end

    # Compute directions wₖ₋₂, ẘₖ₋₁ and w̄ₖ, last columns of Wₖ = Vₖ(Pₖ)ᵀ
    if iter == 1
      # w̅₁ = v₁
      @. wₖ = vₖ
    elseif iter == 2
      # [w̅ₖ₋₁ vₖ] [cpₖ  spₖ] = [ẘₖ₋₁ w̅ₖ] ⟷ ẘₖ₋₁ = cpₖ * w̅ₖ₋₁ + spₖ * vₖ
      #           [spₖ -cpₖ]             ⟷ w̅ₖ   = spₖ * w̅ₖ₋₁ - cpₖ * vₖ
      @kswap(wₖ₋₁, wₖ)
      @. wₖ = spₖ * wₖ₋₁ - cpₖ * vₖ
      @kaxpby!(n, spₖ, vₖ, cpₖ, wₖ₋₁)
    else
      # [ẘₖ₋₂ w̄ₖ₋₁ vₖ] [cpₖ  0   spₖ] [1   0    0 ] = [wₖ₋₂ ẘₖ₋₁ w̄ₖ] ⟷ wₖ₋₂ = cpₖ * ẘₖ₋₂ + spₖ * vₖ
      #                [ 0   1    0 ] [0  cdₖ  sdₖ]                  ⟷ ẘₖ₋₁ = cdₖ * w̄ₖ₋₁ + sdₖ * (spₖ * ẘₖ₋₂ - cpₖ * vₖ)
      #                [spₖ  0  -cpₖ] [0  sdₖ -cdₖ]                  ⟷ w̄ₖ   = sdₖ * w̄ₖ₋₁ - cdₖ * (spₖ * ẘₖ₋₂ - cpₖ * vₖ)
      ẘₖ₋₂ = wₖ₋₁
      w̄ₖ₋₁ = wₖ
      # Update the solution x
      @kaxpy!(n, cpₖ * τₖ₋₂, ẘₖ₋₂, x)
      @kaxpy!(n, spₖ * τₖ₋₂, vₖ, x)
      # Compute wₐᵤₓ = spₖ * ẘₖ₋₂ - cpₖ * vₖ
      @kaxpby!(n, -cpₖ, vₖ, spₖ, ẘₖ₋₂)
      wₐᵤₓ = ẘₖ₋₂
      # Compute ẘₖ₋₁ and w̄ₖ
      @kref!(n, w̄ₖ₋₁, wₐᵤₓ, cdₖ, sdₖ)
      @kswap(wₖ₋₁, wₖ)
    end

    # Update Lanczos vectors
    @. vₖ₋₁ = vₖ
    if βₖ₊₁ ≠ zero(T)
      @. vₖ = p / βₖ₊₁
    end

    # Update ‖rₖ‖ estimate
    # ‖ rₖ ‖ = |ζbarₖ₊₁|
    rNorm = abs(ζbarₖ₊₁)
    push!(rNorms, rNorm)

    # Update ‖Arₖ₋₁‖ estimate
    # ‖ Arₖ₋₁ ‖ = |ζbarₖ| * √((λbarₖ)² + (γbarₖ)²)
    ArNorm = abs(ζbarₖ) * √(λbarₖ^2 + (cₖ₋₁ * βₖ₊₁)^2)
    push!(ArNorms, ArNorm)

    # Update stopping criterion.
    iter == 1 && (κ = (atol + rtol * ArNorm) / 100)
    solved = rNorm ≤ ε
    inconsistent = !solved && ArNorm ≤ κ
    tired = iter ≥ itmax

    # Update variables
    if iter ≥ 2
      sₖ₋₂ = sₖ₋₁
      cₖ₋₂ = cₖ₋₁
      ξₖ₋₁ = ξₖ
      μbisₖ₋₂ = μbisₖ₋₁
      ψbarₖ₋₂ = ψbarₖ₋₁
    end
    sₖ₋₁ = sₖ
    cₖ₋₁ = cₖ
    μbarₖ₋₁ = μbarₖ
    ζbarₖ = ζbarₖ₊₁
    βₖ = βₖ₊₁
    verbose && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm, ArNorm)
  end
  verbose && @printf("\n")

  # Finalize the update of x
  if iter ≥ 2
    @kaxpy!(n, τₖ₋₁, wₖ₋₁, x)
  end
  if !inconsistent
    @kaxpy!(n, τₖ, wₖ, x)
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, stats)
end
