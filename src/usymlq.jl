# An implementation of USYMLQ for the solution of linear system Ax = b.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point and quasi-definite systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, November 2018.

export usymlq, usymlq!

"""
    (x, stats) = usymlq(A, b::AbstractVector{T}, c::AbstractVector{T};
                        atol::T=√eps(T), rtol::T=√eps(T), transfer_to_usymcg::Bool=true,
                        itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the linear system Ax = b using the USYMLQ method.

USYMLQ is based on a tridiagonalization process for unsymmetric matrices.
The error norm ‖x - x*‖ monotonously decreases in USYMLQ.
It's considered as a generalization of SYMMLQ.

It can also be applied to under-determined and over-determined problems.
In all cases, problems must be consistent.

An option gives the possibility of transferring to the USYMCG point,
when it exists. The transfer is based on the residual norm.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, *Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*, SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, *A tridiagonalization method for symmetric saddle-point and quasi-definite systems*, SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, *BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*, SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function usymlq(A, b :: AbstractVector{T}, c :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = UsymlqSolver(A, b)
  usymlq!(solver, A, b, c; kwargs...)
end

function usymlq!(solver :: UsymlqSolver{T,S}, A, b :: AbstractVector{T}, c :: AbstractVector{T};
                 atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_usymcg :: Bool=true,
                 itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("USYMLQ: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  uₖ₋₁, uₖ, x, d̅, vₖ₋₁, vₖ = solver.uₖ₋₁, solver.uₖ, solver.x, solver.d̅, solver.vₖ₋₁, solver.vₖ

  # Initial solution x₀ and residual norm ‖r₀‖.
  x .= zero(T)
  bNorm = @knrm2(m, b)
  bNorm == 0 && return (x, SimpleStats(true, false, [bNorm], T[], "x = 0 is a zero-residual solution"))

  iter = 0
  itmax == 0 && (itmax = m+n)

  rNorms = history ? [bNorm] : T[]
  ε = atol + rtol * bNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, bNorm)

  βₖ = @knrm2(m, b)          # β₁ = ‖v₁‖
  γₖ = @knrm2(n, c)          # γ₁ = ‖u₁‖
  vₖ₋₁ .= zero(T)            # v₀ = 0
  uₖ₋₁ .= zero(T)            # u₀ = 0
  vₖ .= b ./ βₖ              # v₁ = b / β₁
  uₖ .= c ./ γₖ              # u₁ = c / γ₁
  cₖ₋₁ = cₖ = -one(T)        # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(T)        # Givens sines used for the LQ factorization of Tₖ
  d̅ .= zero(T)               # Last column of D̅ₖ = Uₖ(Qₖ)ᵀ
  ζₖ₋₁ = ζbarₖ = zero(T)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(T)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(T)  # Coefficients of Lₖ₋₁ and Lₖ modified over the course of two iterations

  # Stopping criterion.
  solved_lq = bNorm ≤ ε
  solved_cg = false
  tired     = iter ≥ itmax
  status    = "unknown"

  while !(solved_lq || solved_cg || tired)
    # Update iteration index.
    iter = iter + 1

    # Continue the SSY tridiagonalization process.
    # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᵀVₖ = Uₖ(Tₖ)ᵀ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    mul!(vₖ₋₁, A , uₖ, one(T), -γₖ)  # Forms vₖ₊₁ : vₖ₋₁ ← Auₖ  - γₖvₖ₋₁
    mul!(uₖ₋₁, Aᵀ, vₖ, one(T), -βₖ)  # Forms uₖ₊₁ : uₖ₋₁ ← Aᵀvₖ - βₖuₖ₋₁

    αₖ = @kdot(m, vₖ, vₖ₋₁)  # αₖ = (Auₖ- γₖvₖ₋₁)ᵀvₖ

    @kaxpy!(m, -αₖ, vₖ, vₖ₋₁)  # vₖ₋₁ ← vₖ₋₁ - αₖ * vₖ
    @kaxpy!(n, -αₖ, uₖ, uₖ₋₁)  # uₖ₋₁ ← uₖ₋₁ - αₖ * uₖ

    βₖ₊₁ = @knrm2(m, vₖ₋₁)  # βₖ₊₁ = ‖vₖ₊₁‖
    γₖ₊₁ = @knrm2(n, uₖ₋₁)  # γₖ₊₁ = ‖uₖ₊₁‖

    # Compute uₖ₊₁ and uₖ₊₁.
    if βₖ₊₁ ≠ zero(T)
      @. vₖ₋₁ = vₖ₋₁ / βₖ₊₁ # βₖ₊₁vₖ₊₁ = Auₖ  - γₖvₖ₋₁ - αₖvₖ
    end
    if γₖ₊₁ ≠ zero(T)
      @. uₖ₋₁ = uₖ₋₁ / γₖ₊₁ # γₖ₊₁uₖ₊₁ = Aᵀvₖ - βₖuₖ₋₁ - αₖuₖ
    end

    # Update the LQ factorization of Tₖ = L̅ₖQₖ.
    # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
    # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
    # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
    # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
    # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
    # [ •        •  •  •  γₖ]   [ •         •   •   •    •    0   ]
    # [ 0  •  •  •  0  βₖ αₖ]   [ •    •    •   0  ϵₖ₋₂ λₖ₋₁ δbarₖ]

    if iter == 1
      δbarₖ = αₖ
    elseif iter == 2
      # [δbar₁ γ₂] [c₂  s₂] = [δ₁   0  ]
      # [ β₂   α₂] [s₂ -c₂]   [λ₁ δbar₂]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      λₖ₋₁  = cₖ * βₖ + sₖ * αₖ
      δbarₖ = sₖ * βₖ - cₖ * αₖ
    else
      # [0  βₖ  αₖ] [cₖ₋₁   sₖ₋₁   0] = [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ]
      #             [sₖ₋₁  -cₖ₋₁   0]
      #             [ 0      0     1]
      #
      # [ λₖ₋₂   δbarₖ₋₁  γₖ] [1   0   0 ] = [λₖ₋₂  δₖ₋₁    0  ]
      # [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ] [0   cₖ  sₖ]   [ϵₖ₋₂  λₖ₋₁  δbarₖ]
      #                       [0   sₖ -cₖ]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      ϵₖ₋₂  =  sₖ₋₁ * βₖ
      λₖ₋₁  = -cₖ₋₁ * cₖ * βₖ + sₖ * αₖ
      δbarₖ = -cₖ₋₁ * sₖ * βₖ - cₖ * αₖ
    end

    # Compute ζₖ₋₁ and ζbarₖ, last components of the solution of Lₖz̅ₖ = β₁e₁
    # [δbar₁] [ζbar₁] = [β₁]
    if iter == 1
      ηₖ = βₖ
    end
    # [δ₁    0  ] [  ζ₁ ] = [β₁]
    # [λ₁  δbar₂] [ζbar₂]   [0 ]
    if iter == 2
      ηₖ₋₁ = ηₖ
      ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
      ηₖ   = -λₖ₋₁ * ζₖ₋₁
    end
    # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
    # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
    #                     [ζbarₖ]
    if iter ≥ 3
      ζₖ₋₂ = ζₖ₋₁
      ηₖ₋₁ = ηₖ
      ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
      ηₖ   = -ϵₖ₋₂ * ζₖ₋₂ - λₖ₋₁ * ζₖ₋₁
    end

    # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Uₖ(Qₖ)ᵀ.
    # [d̅ₖ₋₁ uₖ] [cₖ  sₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * uₖ
    #           [sₖ -cₖ]             ⟷ d̅ₖ   = sₖ * d̅ₖ₋₁ - cₖ * uₖ
    if iter ≥ 2
      # Compute solution xₖ.
      # (xᴸ)ₖ₋₁ ← (xᴸ)ₖ₋₂ + ζₖ₋₁ * dₖ₋₁
      @kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
      @kaxpy!(n, ζₖ₋₁ * sₖ, uₖ, x)
    end

    # Compute d̅ₖ.
    if iter == 1
      # d̅₁ = u₁
      @. d̅ = uₖ
    else
      # d̅ₖ = sₖ * d̅ₖ₋₁ - cₖ * uₖ
      @kaxpby!(n, -cₖ, uₖ, sₖ, d̅)
    end

    # Update uₖ₋₁, vₖ₋₁, uₖ and vₖ.
    @kswap(uₖ₋₁, uₖ)
    @kswap(vₖ₋₁, vₖ)

    # Compute USYMLQ residual norm
    # ‖rₖ‖ = √((μₖ)² + (ωₖ)²)
    if iter == 1
      rNorm_lq = bNorm
    else
      μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
      ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
      rNorm_lq = sqrt(μₖ^2 + ωₖ^2)
    end
    history && push!(rNorms, rNorm_lq)

    # Compute USYMCG residual norm
    # ‖rₖ‖ = |ρₖ|
    if transfer_to_usymcg && (δbarₖ ≠ 0)
      ζbarₖ = ηₖ / δbarₖ
      ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
      rNorm_cg = abs(ρₖ)
    end

    # Update sₖ₋₁, cₖ₋₁, γₖ, βₖ and δbarₖ₋₁.
    sₖ₋₁    = sₖ
    cₖ₋₁    = cₖ
    γₖ      = γₖ₊₁
    βₖ      = βₖ₊₁
    δbarₖ₋₁ = δbarₖ

    # Update stopping criterion.
    solved_lq = rNorm_lq ≤ ε
    solved_cg = transfer_to_usymcg && (δbarₖ ≠ 0) && (rNorm_cg ≤ ε)
    tired = iter ≥ itmax
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm_lq)
  end
  (verbose > 0) && @printf("\n")

  # Compute USYMCG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

  tired     && (status = "maximum number of iterations exceeded")
  solved_lq && (status = "solution xᴸ good enough given atol and rtol")
  solved_cg && (status = "solution xᶜ good enough given atol and rtol")
  stats = SimpleStats(solved_lq || solved_cg, false, rNorms, T[], status)
  return (x, stats)
end
