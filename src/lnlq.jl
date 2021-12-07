# An implementation of LNLQ for the solution of the consistent linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  Ax = b,
#
# and is equivalent to applying the SYMMLQ method
# to the linear system
#
#  AAᵀy = b with x = Aᵀy and can be reformulated as
#
#  [ -I  Aᵀ ][ x ] = [ 0 ]
#  [  A     ][ y ]   [ b ].
#
# This method is based on the Golub-Kahan bidiagonalization process and is described in
#
# R. Estrin, D. Orban, M.A. Saunders, LNLQ: An Iterative Method for Least-Norm Problems with an Error Minimization Property,
# SIAM Journal on Matrix Analysis and Applications, 40(3), pp. 1102--1124, 2019.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, March 2019 -- Alès, January 2020.

export lnlq, lnlq!

"""
    (x, y, stats) = lnlq(A, b::AbstractVector{FC};
                         M=I, N=I, sqd::Bool=false, λ::T=zero(T), σ::T=zero(T),
                         atol::T=√eps(T), rtol::T=√eps(T), etolx::T=√eps(T), etoly::T=√eps(T), itmax::Int=0,
                         transfer_to_craig::Bool=true, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Find the least-norm solution of the consistent linear system

    Ax + λs = b

using the LNLQ method, where λ ≥ 0 is a regularization parameter.

For a system in the form Ax = b, LNLQ method is equivalent to applying
SYMMLQ to AAᵀy = b and recovering x = Aᵀy but is more stable.
Note that y are the Lagrange multipliers of the least-norm problem

    minimize ‖x‖  s.t.  Ax = b.

If `sqd = true`, LNLQ solves the symmetric and quasi-definite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
LNLQ is then equivalent to applying SYMMLQ to `(AF⁻¹Aᵀ + E)y = b` with `Fx = Aᵀy`.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.

If `sqd = false`, LNLQ solves the symmetric and indefinite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

In this case, M can still be specified and indicates the weighted norm in which residuals are measured.

In this implementation, both the x and y-parts of the solution are returned.

`etolx` and `etoly` are tolerances on the upper bound of the distance to the solution ‖x-xₛ‖ and ‖y-yₛ‖, respectively.
The bound is valid if λ>0 or σ>0 where σ should be strictly smaller than the smallest positive singular value.
For instance σ:=(1-1e-7)σₘᵢₙ .

#### Reference

* R. Estrin, D. Orban, M.A. Saunders, [*LNLQ: An Iterative Method for Least-Norm Problems with an Error Minimization Property*](https://doi.org/10.1137/18M1194948), SIAM Journal on Matrix Analysis and Applications, 40(3), pp. 1102--1124, 2019.
"""
function lnlq(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = LnlqSolver(A, b)
  lnlq!(solver, A, b; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

function lnlq!(solver :: LnlqSolver{T,FC,S}, A, b :: AbstractVector{FC};
               M=I, N=I, sqd :: Bool=false, λ :: T=zero(T), σ :: T=zero(T),
               atol :: T=√eps(T), rtol :: T=√eps(T), etolx :: T=√eps(T), etoly :: T=√eps(T), itmax :: Int=0,
               transfer_to_craig :: Bool=true, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("LNLQ: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₘ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")
  NisI || (promote_type(eltype(N), T) == T) || error("eltype(N) can't be promoted to $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # When solving a SQD system, set regularization parameter λ = 1.
  sqd && (λ = one(T))

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  allocate_if(λ > 0, solver, :q, S, n)
  x, Nv, Aᵀu, y, w̄ = solver.x, solver.Nv, solver.Aᵀu, solver.y, solver.w̄
  Mu, Av, q, stats = solver.Mu, solver.Av, solver.q, solver.stats
  rNorms, xNorms, yNorms = stats.residuals, stats.error_bnd_x, stats.error_bnd_y
  reset!(stats)
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  # Set up parameter σₑₛₜ for the error estimate on x and y
  σₑₛₜ = √(σ^2 + λ^2)
  complex_error_bnd = false

  # Initial solutions (x₀, y₀) and residual norm ‖r₀‖.
  x .= zero(T)
  y .= zero(T)

  bNorm = @knrm2(m, b)
  if bNorm == 0
    stats.solved = true
    stats.error_with_bnd = false
    history && push!(rNorms, bNorm)
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  history && push!(rNorms, bNorm)
  ε = atol + rtol * bNorm

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, bNorm)

  # Update iteration index
  iter = iter + 1

  # Initialize generalized Golub-Kahan bidiagonalization.
  # β₁Mu₁ = b.
  Mu .= b
  MisI || mul!(u, M, Mu)      # u₁ = M⁻¹ * Mu₁
  βₖ = sqrt(@kdot(m, u, Mu))  # β₁ = ‖u₁‖_M
  if βₖ ≠ 0
    @kscal!(m, 1 / βₖ, u)
    MisI || @kscal!(m, 1 / βₖ, Mu)
  end

  # α₁Nv₁ = Aᵀu₁.
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)      # v₁ = N⁻¹ * Nv₁
  αₖ = sqrt(@kdot(n, v, Nv))  # α₁ = ‖v₁‖_N
  if αₖ ≠ 0
    @kscal!(n, 1 / αₖ, v)
    NisI || @kscal!(n, 1 / αₖ, Nv)
  end

  w̄ .= u             # Direction w̄₁
  cₖ = sₖ = zero(T)  # Givens sines and cosines used for the LQ factorization of (Lₖ)ᵀ
  ζₖ₋₁ = zero(T)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ
  ηₖ = zero(T)       # Coefficient of M̅ₖ

  # Variable used for the regularization.
  λₖ  = λ             # λ₁ = λ
  cpₖ = spₖ = one(T)  # Givens sines and cosines used to zero out λₖ
  cdₖ = sdₖ = one(T)  # Givens sines and cosines used to define λₖ₊₁
  λ > 0 && (q .= v)   # Additional vector needed to update x, by definition q₀ = 0

  # Initialize the regularization.
  if λ > 0
    #        k    2k      k   2k           k      2k
    # k   [  αₖ   λₖ ] [ cpₖ  spₖ ] = [  αhatₖ    0   ]
    # k+1 [ βₖ₊₁  0  ] [ spₖ -cpₖ ]   [ βhatₖ₊₁  θₖ₊₁ ]
    (cpₖ, spₖ, αhatₖ) = sym_givens(αₖ, λₖ)

    # q̄₁ = sp₁ * v₁
    @kscal!(n, spₖ, q)
  else
    αhatₖ = αₖ
  end

  # Begin the LQ factorization of (Lₖ)ᵀ = M̅ₖQₖ.
  # [ α₁ β₂ 0  •  •  •  0 ]   [ ϵ₁  0   •   •   •   •   0   ]
  # [ 0  α₂ •  •        • ]   [ η₂  ϵ₂  •               •   ]
  # [ •  •  •  •  •     • ]   [ 0   •   •   •           •   ]
  # [ •     •  •  •  •  • ] = [ •   •   •   •   •       •   ] Qₖ
  # [ •        •  •  •  0 ]   [ •       •   •   •   •   •   ]
  # [ •           •  •  βₖ]   [ •           •   •   •   0   ]
  # [ 0  •  •  •  •  0  αₖ]   [ 0   •   •   •   0   ηₖ ϵbarₖ]

  ϵbarₖ = αhatₖ  # ϵbar₁ = αhat₁

  # Hₖ = Bₖ(Lₖ)ᵀ = [   Lₖ(Lₖ)ᵀ   ] ⟹ (Hₖ₋₁)ᵀ = [Lₖ₋₁Mₖ₋₁  0] Qₖ
  #                [ αₖβₖ₊₁(eₖ)ᵀ ]
  #
  # Solve Lₖtₖ = β₁e₁ and M̅ₖz̅ₖ = tₖ
  # tₖ = (τ₁, •••, τₖ)
  # z̅ₖ = (zₖ₋₁, ζbarₖ) = (ζ₁, •••, ζₖ₋₁, ζbarₖ)

  τₖ    = βₖ / αhatₖ  # τ₁ = β₁ / αhat₁
  ζbarₖ = τₖ / ϵbarₖ  # ζbar₁ = τ₁ / ϵbar₁

  # Stopping criterion.
  solved_lq = solved_cg = false
  tired = false
  status = "unknown"

  if σₑₛₜ > 0
    τtildeₖ = βₖ / σₑₛₜ
    ζtildeₖ = τtildeₖ / σₑₛₜ
    err_x = τtildeₖ
    err_y = ζtildeₖ

    solved_lq = err_x ≤ etolx || err_y ≤ etoly
    history && push!(xNorms, err_x)
    history && push!(yNorms, err_y)

    ρbar = -σₑₛₜ
    csig = -one(T)
  end

  while !(solved_lq || solved_cg || tired)

    # Update of (xᵃᵘˣ)ₖ = Vₖtₖ
    if λ > 0
      # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, τₖ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, τₖ * spₖ, q, x)
        # q̄ₖ ← spₖ * vₖ - cpₖ * qₖ₋₁
        @kaxpby!(n, spₖ, v, -cpₖ, q)
      end
    else
      # (xᵃᵘˣ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
      @kaxpy!(n, τₖ, v, x)
    end

    # Continue the generalized Golub-Kahan bidiagonalization.
    # AVₖ    = MUₖ₊₁Bₖ
    # AᵀUₖ₊₁ = NVₖ(Bₖ)ᵀ + αₖ₊₁Nvₖ₊₁(eₖ₊₁)ᵀ = NVₖ₊₁(Lₖ₊₁)ᵀ
    #
    #      [ α₁ 0  •  •  •  •  0 ]
    #      [ β₂ α₂ •           • ]
    #      [ 0  •  •  •        • ]
    # Lₖ = [ •  •  •  •  •     • ]
    #      [ •     •  •  •  •  • ]
    #      [ •        •  •  •  0 ]
    #      [ 0  •  •  •  0  βₖ αₖ]
    #
    # Bₖ = [    Lₖ     ]
    #      [ βₖ₊₁(eₖ)ᵀ ]

    # βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(T), Av, -αₖ, Mu)
    MisI || mul!(u, M, Mu)        # uₖ₊₁ = M⁻¹ * Muₖ₊₁
    βₖ₊₁ = sqrt(@kdot(m, u, Mu))  # βₖ₊₁ = ‖uₖ₊₁‖_M
    if βₖ₊₁ ≠ 0
      @kscal!(m, 1 / βₖ₊₁, u)
      MisI || @kscal!(m, 1 / βₖ₊₁, Mu)
    end

    # αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    mul!(Aᵀu, Aᵀ, u)
    @kaxpby!(n, one(T), Aᵀu, -βₖ₊₁, Nv)
    NisI || mul!(v, N, Nv)        # vₖ₊₁ = N⁻¹ * Nvₖ₊₁
    αₖ₊₁ = sqrt(@kdot(n, v, Nv))  # αₖ₊₁ = ‖vₖ₊₁‖_N
    if αₖ₊₁ ≠ 0
      @kscal!(n, 1 / αₖ₊₁, v)
      NisI || @kscal!(n, 1 / αₖ₊₁, Nv)
    end

    # Continue the regularization.
    if λ > 0
      #        k    2k      k   2k           k      2k
      # k   [  αₖ   λₖ ] [ cpₖ  spₖ ] = [  αhatₖ    0   ]
      # k+1 [ βₖ₊₁  0  ] [ spₖ -cpₖ ]   [ βhatₖ₊₁  θₖ₊₁ ]
      βhatₖ₊₁ = cpₖ * βₖ₊₁
      θₖ₊₁    = spₖ * βₖ₊₁

      #       2k  2k+1     2k  2k+1       2k  2k+1
      # k   [  0    0 ] [ -cdₖ  sdₖ ] = [ 0    0  ]
      # k+1 [ θₖ₊₁  λ ] [  sdₖ  cdₖ ]   [ 0  λₖ₊₁ ]
      (cdₖ, sdₖ, λₖ₊₁) = sym_givens(λ, θₖ₊₁)

      # qₖ ← sdₖ * q̄ₖ
      @kscal!(n, sdₖ, q)

      #       k+1   2k+1      k+1    2k+1        k+1     2k+1
      # k+1 [ αₖ₊₁  λₖ₊₁ ] [ cpₖ₊₁  spₖ₊₁ ] = [ αhatₖ₊₁   0   ]
      # k+2 [ βₖ₊₂   0   ] [ spₖ₊₁ -cpₖ₊₁ ]   [  γₖ₊₂    θₖ₊₂ ]
      (cpₖ₊₁, spₖ₊₁, αhatₖ₊₁) = sym_givens(αₖ₊₁, λₖ₊₁)
    else
      βhatₖ₊₁ = βₖ₊₁
      αhatₖ₊₁ = αₖ₊₁
    end

    if σₑₛₜ > 0 && !complex_error_bnd
      μbar = -csig * αhatₖ
      ρ = √(ρbar^2 + αhatₖ^2)
      csig = ρbar / ρ
      ssig = αhatₖ / ρ
      ρbar = ssig * μbar + csig * σₑₛₜ
      μbar = -csig * βhatₖ₊₁
      θ = βhatₖ₊₁ * csig / ρbar
      ωdisc = σₑₛₜ^2 - σₑₛₜ * βhatₖ₊₁ * θ
      if ωdisc < 0
        complex_error_bnd = true
      else
        ω = √ωdisc
        τtildeₖ = - τₖ * βhatₖ₊₁ / ω
      end

      ρ = √(ρbar^2 + βhatₖ₊₁^2)
      csig = ρbar / ρ
      ssig = βhatₖ₊₁ / ρ
      ρbar = ssig * μbar + csig * σₑₛₜ
    end

    # Continue the LQ factorization of (Lₖ₊₁)ᵀ.
    # [ηₖ ϵbarₖ βₖ₊₁] [1     0     0 ] = [ηₖ  ϵₖ     0    ]
    # [0    0   αₖ₊₁] [0   cₖ₊₁  sₖ₊₁]   [0  ηₖ₊₁  ϵbarₖ₊₁]
    #                 [0   sₖ₊₁ -cₖ₊₁]

    (cₖ₊₁, sₖ₊₁, ϵₖ) = sym_givens(ϵbarₖ, βhatₖ₊₁)
    ηₖ₊₁    =   αhatₖ₊₁ * sₖ₊₁
    ϵbarₖ₊₁ = - αhatₖ₊₁ * cₖ₊₁

    # Update solutions of Lₖ₊₁tₖ₊₁ = β₁e₁ and M̅ₖ₊₁z̅ₖ₊₁ = tₖ₊₁.
    τₖ₊₁    = - βhatₖ₊₁ * τₖ / αhatₖ₊₁
    ζₖ      = cₖ₊₁ * ζbarₖ
    ζbarₖ₊₁ = (τₖ₊₁ - ηₖ₊₁ * ζₖ) / ϵbarₖ₊₁

    # Relations for the directions wₖ and w̄ₖ₊₁
    # [w̄ₖ uₖ₊₁] [cₖ₊₁  sₖ₊₁] = [wₖ w̄ₖ₊₁] → wₖ   = cₖ₊₁ * w̄ₖ + sₖ₊₁ * uₖ₊₁
    #           [sₖ₊₁ -cₖ₊₁]             → w̄ₖ₊₁ = sₖ₊₁ * w̄ₖ - cₖ₊₁ * uₖ₊₁

    # (yᴸ)ₖ₊₁ ← (yᴸ)ₖ + ζₖ * wₖ
    @kaxpy!(m, ζₖ * cₖ₊₁, w̄, y)
    @kaxpy!(m, ζₖ * sₖ₊₁, u, y)

    # Compute w̄ₖ₊₁
    @kaxpby!(m, -cₖ₊₁, u, sₖ₊₁, w̄)

    if σₑₛₜ > 0 && !complex_error_bnd
      if transfer_to_craig
        disc_x = τtildeₖ^2 - τₖ₊₁^2
        disc_x < 0 ? complex_error_bnd = true : err_x = √disc_x
      else
        disc_xL = τtildeₖ^2 - τₖ₊₁^2 + (τₖ₊₁ - ηₖ₊₁ * ζₖ)^2
        disc_xL < 0 ? complex_error_bnd = true : err_x = √disc_xL
      end
      ηtildeₖ = ω * sₖ₊₁
      ϵtildeₖ = -ω * cₖ₊₁
      ζtildeₖ = (τtildeₖ - ηtildeₖ * ζₖ) / ϵtildeₖ
      
      if transfer_to_craig
        disc_y = ζtildeₖ^2 - ζbarₖ₊₁^2
        disc_y < 0 ? complex_error_bnd = true : err_y = √disc_y
      else
        err_y = abs(ζtildeₖ)
      end

      history && push!(xNorms, err_x)
      history && push!(yNorms, err_y)
    end

    # Compute residual norm ‖(rᴸ)ₖ‖ = |αₖ| * √((ϵbarₖζbarₖ)² + (βₖ₊₁sₖζₖ₋₁)²)
    if iter == 1
      rNorm_lq = bNorm
    else
      rNorm_lq = abs(αhatₖ) * √((ϵbarₖ * ζbarₖ)^2 + (βhatₖ₊₁ * sₖ * ζₖ₋₁)^2)
    end
    history && push!(rNorms, rNorm_lq)

    # Compute residual norm ‖(rᶜ)ₖ‖ = |βₖ₊₁ * τₖ|
    if transfer_to_craig
      rNorm_cg = abs(βhatₖ₊₁ * τₖ)
    end

    # Update sₖ, cₖ, αₖ, βₖ, ηₖ, ϵbarₖ, τₖ, ζₖ₋₁ and ζbarₖ.
    cₖ    = cₖ₊₁
    sₖ    = sₖ₊₁
    αₖ    = αₖ₊₁
    αhatₖ = αhatₖ₊₁
    βₖ    = βₖ₊₁
    ηₖ    = ηₖ₊₁
    ϵbarₖ = ϵbarₖ₊₁
    τₖ    = τₖ₊₁
    ζₖ₋₁  = ζₖ
    ζbarₖ = ζbarₖ₊₁

    # Update regularization variables.
    if λ > 0
      cpₖ = cpₖ₊₁
      spₖ = spₖ₊₁
    end

    # Update stopping criterion.
    tired = iter ≥ itmax
    solved_lq = rNorm_lq ≤ ε
    solved_cg = transfer_to_craig && rNorm_cg ≤ ε
    if σₑₛₜ > 0
      if transfer_to_craig
        solved_cg = solved_cg || err_x ≤ etolx || err_y ≤ etoly
      else
        solved_lq = solved_lq || err_x ≤ etolx || err_y ≤ etoly
      end
    end
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm_lq)

    # Update iteration index.
    iter = iter + 1
  end
  (verbose > 0) && @printf("\n")

  if solved_cg
    if λ > 0
      # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, τₖ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, τₖ * spₖ, q, x)
      end
    else
      # (xᶜ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + τₖ * vₖ
      @kaxpy!(n, τₖ, v, x)
    end
    # (yᶜ)ₖ ← (yᴸ)ₖ₋₁ + ζbarₖ * w̄ₖ
    @kaxpy!(m, ζbarₖ, w̄, y)
  else
    if λ > 0
      # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * (cpₖvₖ + spₖqₖ₋₁)
      @kaxpy!(n, ηₖ * ζₖ₋₁ * cpₖ, v, x)
      if iter ≥ 2
        @kaxpy!(n, ηₖ * ζₖ₋₁ * spₖ, q, x)
      end
    else
      # (xᴸ)ₖ ← (xᵃᵘˣ)ₖ₋₁ + ηₖζₖ₋₁ * vₖ
      @kaxpy!(n, ηₖ * ζₖ₋₁, v, x)
    end
  end

  tired     && (status = "maximum number of iterations exceeded")
  solved_lq && (status = "solutions (xᴸ, yᴸ) good enough for the tolerances given")
  solved_cg && (status = "solutions (xᶜ, yᶜ) good enough for the tolerances given")

  # Update stats
  stats.solved = solved_lq || solved_cg
  stats.error_with_bnd = complex_error_bnd
  stats.status = status
  return solver
end
