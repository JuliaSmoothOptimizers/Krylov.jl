# An implementation of BILQR for the solution of square
# consistent linear adjoint systems Ax = b and Aᵀt = c.
#
# This method is described in
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# Cahier du GERAD G-2019-71, GERAD, Montreal, 2019. doi:10.13140/RG.2.2.18287.59042
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, July 2019.

export bilqr

"""
    (x, t, stats) = bilqr(A, b, c; atol, rtol, transfer_to_bicg, itmax, verbose)

Combine BiLQ and QMR to solve adjoint systems.

    [0  A] [t] = [b]
    [Aᵀ 0] [x]   [c]

BiLQ is used for solving primal system `Ax = b`.
QMR is used for solving dual system `Aᵀt = c`.

An option gives the possibility of transferring from the BiLQ point to the
BiCG point, when it exists. The transfer is based on the residual norm.
"""
function bilqr(A, b :: AbstractVector{T}, c :: AbstractVector{T};
               atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_bicg :: Bool=true,
               itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n, m = size(A)
  m == n || error("Systems must be square")
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  verbose && @printf("BILQR: systems of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # Determine the storage type of b
  S = typeof(b)

  # Initial solution x₀ and residual norm ‖r₀‖ = ‖b - Ax₀‖.
  x = kzeros(S, n)      # x₀
  bNorm = @knrm2(n, b)  # rNorm = ‖r₀‖

  # Initial solution t₀ and residual norm ‖s₀‖ = ‖c - Aᵀt₀‖.
  t = kzeros(S, n)      # t₀
  cNorm = @knrm2(n, c)  # sNorm = ‖s₀‖

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [bNorm;]
  sNorms = [cNorm;]
  εL = atol + rtol * bNorm
  εQ = atol + rtol * cNorm
  verbose && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖sₖ‖")
  verbose && @printf("%5d  %7.1e  %7.1e\n", iter, bNorm, cNorm)

  # Initialize the Lanczos biorthogonalization process.
  bᵗc = @kdot(n, b, c)  # ⟨b,c⟩
  bᵗc == 0 && return (x, y, AdjointStats(false, false, [bNorm], [cNorm], "Breakdown bᵀc = 0"))

  # Set up workspace.
  βₖ = √(abs(bᵗc))           # β₁γ₁ = bᵀc
  γₖ = bᵗc / βₖ              # β₁γ₁ = bᵀc
  vₖ₋₁ = kzeros(S, n)        # v₀ = 0
  uₖ₋₁ = kzeros(S, n)        # u₀ = 0
  vₖ = b / βₖ                # v₁ = b / β₁
  uₖ = c / γₖ                # u₁ = c / γ₁
  cₖ₋₁ = cₖ = -one(T)        # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(T)        # Givens sines used for the LQ factorization of Tₖ
  d̅ = kzeros(S, n)           # Last column of D̅ₖ = Vₖ(Qₖ)ᵀ
  ζₖ₋₁ = ζbarₖ = zero(T)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(T)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(T)  # Coefficients of Lₖ₋₁ and L̅ₖ modified over the course of two iterations
  ψbarₖ₋₁ = ψₖ₋₁ = zero(T)   # ψₖ₋₁ and ψbarₖ are the last components of h̅ₖ = Qₖγ₁e₁
  norm_vₖ = bNorm / βₖ       # ‖vₖ‖ is used for residual norm estimates
  ϵₖ₋₃ = λₖ₋₂ = zero(T)      # Components of Lₖ₋₁
  wₖ₋₃ = kzeros(S, n)        # Column k-3 of Wₖ = Uₖ(Lₖ)⁻ᵀ
  wₖ₋₂ = kzeros(S, n)        # Column k-2 of Wₖ = Uₖ(Lₖ)⁻ᵀ
  τₖ = zero(T)               # τₖ is used for the dual residual norm estimate

  # Stopping criterion.
  solved_lq = bNorm == 0
  solved_lq_tol = solved_lq_mach = false
  solved_cg = solved_cg_tol = solved_cg_mach = false
  solved_primal = solved_lq || solved_cg
  solved_qr_tol = solved_qr_mach = false
  solved_dual = cNorm == 0
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"

  while !((solved_primal && solved_dual) || tired || breakdown)
    # Update iteration index.
    iter = iter + 1

    # Continue the Lanczos biorthogonalization process.
    # AVₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᵀUₖ = Uₖ(Tₖ)ᵀ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    q = A  * vₖ  # Forms vₖ₊₁ : q ← Avₖ
    p = Aᵀ * uₖ  # Forms uₖ₊₁ : p ← Aᵀuₖ

    @kaxpy!(n, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
    @kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - βₖ * uₖ₋₁

    αₖ = @kdot(n, q, uₖ)      # αₖ = qᵀuₖ

    @kaxpy!(n, -αₖ, vₖ, q)    # q ← q - αₖ * vₖ
    @kaxpy!(n, -αₖ, uₖ, p)    # p ← p - αₖ * uₖ

    qᵗp = @kdot(n, p, q)      # qᵗp  = ⟨q,p⟩
    βₖ₊₁ = √(abs(qᵗp))        # βₖ₊₁ = √(|qᵗp|)
    γₖ₊₁ = qᵗp / βₖ₊₁         # γₖ₊₁ = qᵗp / βₖ₊₁

    # Update the LQ factorization of Tₖ = L̅ₖQₖ.
    # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
    # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
    # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
    # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
    # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
    # [ •        •  •  •  γₖ]   [ •         •   •  λₖ₋₂ δₖ₋₁  0   ]
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

    if !solved_primal
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

      # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Vₖ(Qₖ)ᵀ.
      # [d̅ₖ₋₁ vₖ] [cₖ  sₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * vₖ
      #           [sₖ -cₖ]             ⟷ d̅ₖ   = sₖ * d̅ₖ₋₁ - cₖ * vₖ
      if iter ≥ 2
        # Compute solution xₖ.
        # (xᴸ)ₖ ← (xᴸ)ₖ₋₁ + ζₖ₋₁ * dₖ₋₁
        @kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
        @kaxpy!(n, ζₖ₋₁ * sₖ, vₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = v₁
        @. d̅ = vₖ
      else
        # d̅ₖ = sₖ * d̅ₖ₋₁ - cₖ * vₖ
        @kaxpby!(n, -cₖ, vₖ, sₖ, d̅)
      end

      # Compute ⟨vₖ,vₖ₊₁⟩ and ‖vₖ₊₁‖
      vₖᵀvₖ₊₁ = @kdot(n, vₖ, q) / βₖ₊₁
      norm_vₖ₊₁ = @knrm2(n, q) / βₖ₊₁

      # Compute BiLQ residual norm
      # ‖rₖ‖ = √((μₖ)²‖vₖ‖² + (ωₖ)²‖vₖ₊₁‖² + 2μₖωₖ⟨vₖ,vₖ₊₁⟩)
      if iter == 1
        rNorm_lq = bNorm
      else
        μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
        ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
        rNorm_lq = sqrt(μₖ^2 * norm_vₖ^2 + ωₖ^2 * norm_vₖ₊₁^2 + 2 * μₖ * ωₖ * vₖᵀvₖ₊₁)
      end
      push!(rNorms, rNorm_lq)

      # Update ‖vₖ‖
      norm_vₖ = norm_vₖ₊₁

      # Compute BiCG residual norm
      # ‖rₖ‖ = |ρₖ| * ‖vₖ₊₁‖
      if transfer_to_bicg && (δbarₖ ≠ 0)
        ζbarₖ = ηₖ / δbarₖ
        ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
        rNorm_cg = abs(ρₖ) * norm_vₖ₊₁
      end

      # Update primal stopping criterion
      solved_lq_tol = rNorm_lq ≤ εL
      solved_lq_mach = rNorm_lq + 1 ≤ 1
      solved_lq = solved_lq_tol || solved_lq_mach
      solved_cg_tol = transfer_to_bicg && (δbarₖ ≠ 0) && (rNorm_cg ≤ εL)
      solved_cg_mach = transfer_to_bicg && (δbarₖ ≠ 0) && (rNorm_cg + 1 ≤ 1)
      solved_cg = solved_cg_tol || solved_cg_mach
      solved_primal = solved_lq || solved_cg
    end

    if !solved_dual
      # Compute ψₖ₋₁ and ψbarₖ.
      if iter == 1
        ψbarₖ = γₖ
      else
        ψₖ₋₁  = cₖ * ψbarₖ₋₁
        ψbarₖ = sₖ * ψbarₖ₋₁
      end

      # Compute the direction wₖ₋₁, the last column of Wₖ₋₁ = (Uₖ₋₁)(Lₖ₋₁)⁻ᵀ ⟷ (Lₖ₋₁)(Wₖ₋₁)ᵀ = (Uₖ₋₁)ᵀ.
      # w₁ = u₁ / δ₁
      if iter == 2
        wₖ₋₁ = wₖ₋₂
        @kaxpy!(n, one(T), uₖ₋₁, wₖ₋₁)
        @. wₖ₋₁ = uₖ₋₁ / δₖ₋₁
      end
      # w₂ = (u₂ - λ₁w₁) / δ₂
      if iter == 3
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(n, one(T), uₖ₋₁, wₖ₋₁)
        @kaxpy!(n, -λₖ₋₂, wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / δₖ₋₁
      end
      # wₖ₋₁ = (uₖ₋₁ - λₖ₋₂wₖ₋₂ - ϵₖ₋₃wₖ₋₃) / δₖ₋₁
      if iter ≥ 4
        @kscal!(n, -ϵₖ₋₃, wₖ₋₃)
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(n, one(T), uₖ₋₁, wₖ₋₁)
        @kaxpy!(n, -λₖ₋₂, wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / δₖ₋₁
      end

      if iter ≥ 3
        # Swap pointers.
        @kswap(wₖ₋₃, wₖ₋₂)
      end

      if iter ≥ 2
        # Compute solution tₖ₋₁.
        # tₖ₋₁ ← tₖ₋₂ + ψₖ₋₁ * wₖ₋₁
        @kaxpy!(n, ψₖ₋₁, wₖ₋₁, t)
      end

      # Update ψbarₖ₋₁
      ψbarₖ₋₁ = ψbarₖ

      # Compute τₖ = τₖ₋₁ + ‖uₖ‖²
      τₖ += @kdot(n, uₖ, uₖ)

      # Compute QMR residual norm ‖sₖ₋₁‖ ≤ |ψbarₖ| * √τₖ
      sNorm = abs(ψbarₖ) * √τₖ
      push!(sNorms, sNorm)

      # Update dual stopping criterion
      solved_qr_tol = sNorm ≤ εQ
      solved_qr_mach = sNorm + 1 ≤ 1
      solved_dual = solved_qr_tol || solved_qr_mach
    end

    # Compute vₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ  # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ  # uₖ₋₁ ← uₖ

    if qᵗp ≠ zero(T)
      @. vₖ = q / βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
      @. uₖ = p / γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
    end

    # Update ϵₖ₋₃, λₖ₋₂, δbarₖ₋₁, cₖ₋₁, sₖ₋₁, γₖ and βₖ.
    if iter ≥ 3
      ϵₖ₋₃ = ϵₖ₋₂
    end
    if iter ≥ 2
      λₖ₋₂ = λₖ₋₁
    end
    δbarₖ₋₁ = δbarₖ
    cₖ₋₁    = cₖ
    sₖ₋₁    = sₖ
    γₖ      = γₖ₊₁
    βₖ      = βₖ₊₁

    tired = iter ≥ itmax
    breakdown = !solved_lq && !solved_cg && (qᵗp == 0)

    verbose &&  solved_primal && !solved_dual && @printf("%5d  %7s  %7.1e\n", iter, "", sNorm)
    verbose && !solved_primal &&  solved_dual && @printf("%5d  %7.1e  %7s\n", iter, rNorm_lq, "")
    verbose && !solved_primal && !solved_dual && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm_lq, sNorm)
  end
  verbose && @printf("\n")

  # Compute BICG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

   tired                            && (status = "maximum number of iterations exceeded")
   breakdown                        && (status = "Breakdown ⟨uₖ₊₁,vₖ₊₁⟩ = 0")
   solved_lq_tol  && !solved_dual   && (status = "Only the primal solution xᴸ is good enough given atol and rtol")
   solved_cg_tol  && !solved_dual   && (status = "Only the primal solution xᶜ is good enough given atol and rtol")
  !solved_primal  && solved_qr_tol  && (status = "Only the dual solution t is good enough given atol and rtol")
   solved_lq_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᴸ, t) are good enough given atol and rtol")
   solved_cg_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol")
   solved_lq_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᴸ")
   solved_cg_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᶜ")
  !solved_primal  && solved_qr_mach && (status = "Only found approximate zero-residual dual solution t")
   solved_lq_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᴸ, t)")
   solved_cg_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᶜ, t)")
   solved_lq_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᴸ and a dual solution t good enough given atol and rtol")
   solved_cg_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᶜ and a dual solution t good enough given atol and rtol")
   solved_lq_tol  && solved_qr_mach && (status = "Found a primal solution xᴸ good enough given atol and rtol and an approximate zero-residual dual solutions t")
   solved_cg_tol  && solved_qr_mach && (status = "Found a primal solution xᶜ good enough given atol and rtol and an approximate zero-residual dual solutions t")
  stats = AdjointStats(solved_primal, solved_dual, rNorms, sNorms, status)
  return (x, t, stats)
end
