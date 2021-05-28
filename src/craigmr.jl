# An implementation of CRAIG-MR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖  s.t.  Ax = b,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AAᵀy = b.
#
# This method is equivalent to CRMR, and is described in
#
# D. Orban and M. Arioli. Iterative Solution of Symmetric Quasi-Definite Linear Systems,
# Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, May 2015.

export craigmr, craigmr!


"""
    (x, y, stats) = craigmr(A, b::AbstractVector{T};
                            M=I, N=I, sqd::Bool=false, λ::T=zero(T), atol::T=√eps(T),
                            rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the consistent linear system

    Ax + λs = b

using the CRAIGMR method, where λ ≥ 0 is a regularization parameter.
This method is equivalent to applying the Conjugate Residuals method
to the normal equations of the second kind

    (AAᵀ + λ²I)y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

    min ‖(x,s)‖₂  s.t.  Ax + λs = b.

If `sqd = true`, CRAIGMR solves the symmetric and quasi-definite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   E  ] [ y ] = [ b ],

where E and F are symmetric and positive definite.
CRAIGMR is then equivalent to applying MINRES to `(AF⁻¹Aᵀ + E)y = b` with `Fx = Aᵀy`.
Preconditioners M = E⁻¹ ≻ 0 and N = F⁻¹ ≻ 0 may be provided in the form of linear operators.

If `sqd = false`, CRAIGMR solves the symmetric and indefinite system

    [ -F   Aᵀ ] [ x ]   [ 0 ]
    [  A   0  ] [ y ] = [ b ].

In this case, M can still be specified and indicates the weighted norm in which residuals are measured.

CRAIGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRMR, though can be slightly more accurate,
and intricate to implement. Both the x- and y-parts of the solution are
returned.

#### References

* D. Orban and M. Arioli. *Iterative Solution of Symmetric Quasi-Definite Linear Systems*, Volume 3 of Spotlights. SIAM, Philadelphia, PA, 2017.
* D. Orban, *The Projected Golub-Kahan Process for Constrained, Linear Least-Squares Problems*. Cahier du GERAD G-2014-15, 2014.
"""
function craigmr(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = CraigmrSolver(A, b)
  craigmr!(solver, A, b; kwargs...)
end

function craigmr!(solver :: CraigmrSolver{T,S}, A, b :: AbstractVector{T};
                  M=I, N=I, sqd :: Bool=false, λ :: T=zero(T), atol :: T=√eps(T),
                  rtol :: T=√eps(T), itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CRAIGMR: system of %d equations in %d variables\n", m, n)

  # Tests M == Iₘ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Compute the adjoint of A
  Aᵀ = A'

  # When solving a SQD system, set regularization parameter λ = 1.
  sqd && (λ = one(T))

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  allocate_if(λ > 0, solver, :q, S, n)
  x, Nv, Aᵀu, d, y, Mu, w, wbar, Av, q = solver.x, solver.Nv, solver.Aᵀu, solver.d, solver.y, solver.Mu, solver.w, solver.wbar, solver.Av, solver.q
  u = MisI ? Mu : solver.u
  v = NisI ? Nv : solver.v

  # Compute y such that AAᵀy = b. Then recover x = Aᵀy.
  x .= zero(T)
  y .= zero(T)
  Mu .= b
  MisI || mul!(u, M, Mu)
  β = sqrt(@kdot(m, u, Mu))
  β == 0 && return (x, y, SimpleStats(true, false, [zero(T)], T[], "x = 0 is a zero-residual solution"))

  # Initialize Golub-Kahan process.
  # β₁Mu₁ = b.
  @kscal!(m, one(T)/β, u)
  MisI || @kscal!(m, one(T)/β, Mu)
  # α₁Nv₁ = Aᵀu₁.
  mul!(Aᵀu, Aᵀ, u)
  Nv .= Aᵀu
  NisI || mul!(v, N, Nv)
  α = sqrt(@kdot(n, v, Nv))
  Anorm² = α * α

  iter = 0
  itmax == 0 && (itmax = m + n)

  (verbose > 0) && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s\n", "Aprod", "‖r‖", "‖Aᵀr‖", "β", "α", "cos", "sin", "‖A‖²")
  display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", 1, β, α, β, α, 0, 1, Anorm²)

  # Aᵀb = 0 so x = 0 is a minimum least-squares solution
  α == 0 && return (x, y, SimpleStats(true, false, [β], [zero(T)], "x = 0 is a minimum least-squares solution"))
  @kscal!(n, one(T)/α, v)
  NisI || @kscal!(n, one(T)/α, Nv)

  # Regularization.
  λₖ  = λ             # λ₁ = λ
  cpₖ = spₖ = one(T)  # Givens sines and cosines used to zero out λₖ
  cdₖ = sdₖ = one(T)  # Givens sines and cosines used to define λₖ₊₁
  λ > 0 && (q .= v)   # Additional vector needed to update x, by definition q₀ = 0

  if λ > 0
    (cpₖ, spₖ, αhat) = sym_givens(α, λₖ)
    @kscal!(n, spₖ, q)  # q̄₁ = sp₁ * v₁
  else
    αhat = α
  end

  # Initialize other constants.
  ζbar = β
  ρbar = αhat
  θ = zero(T)
  rNorm = ζbar
  rNorms = history ? [rNorm] : T[]
  ArNorm = α
  ArNorms = history ? [ArNorm] : T[]

  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.

  wbar .= u
  @kscal!(m, one(T)/αhat, wbar)
  w .= zero(T)
  d .= zero(T)

  status = "unknown"
  solved = rNorm ≤ ɛ_c
  inconsistent = (rNorm > 100 * ɛ_c) & (ArNorm ≤ ɛ_i)
  tired  = iter ≥ itmax

  while ! (solved || inconsistent || tired)
    iter = iter + 1

    # Generate next Golub-Kahan vectors.
    # 1. βₖ₊₁Muₖ₊₁ = Avₖ - αₖMuₖ
    mul!(Av, A, v)
    @kaxpby!(m, one(T), Av, -α, Mu)
    MisI || mul!(u, M, Mu)
    β = sqrt(@kdot(m, u, Mu))
    if β ≠ 0
      @kscal!(m, one(T)/β, u)
      MisI || @kscal!(m, one(T)/β, Mu)
    end

    Anorm² = Anorm² + β * β  # = ‖B_{k-1}‖²

    if λ > 0
      βhat = cpₖ * β
      λₐᵤₓ = spₖ * β
    else
      βhat = β
    end

    # Continue QR factorization
    #
    # Q [ Lₖ  β₁ e₁ ] = [ Rₖ   zₖ  ] :
    #   [ β    0    ]   [ 0   ζbar ]
    #
    #       k  k+1    k    k+1      k  k+1
    # k   [ c   s ] [ ρbar    ] = [ ρ  θ⁺    ]
    # k+1 [ s  -c ] [ β    α⁺ ]   [    ρbar⁺ ]
    #
    # so that we obtain
    #
    # [ c  s ] [ ζbar ] = [ ζ     ]
    # [ s -c ] [  0   ]   [ ζbar⁺ ]
    (c, s, ρ) = sym_givens(ρbar, βhat)
    ζ = c * ζbar
    ζbar = s * ζbar
    rNorm = abs(ζbar)
    history && push!(rNorms, rNorm)

    @kaxpby!(m, one(T)/ρ, wbar, -θ/ρ, w)  # w = (wbar - θ * w) / ρ
    @kaxpy!(m, ζ, w, y)                   # y = y + ζ * w

    if λ > 0
      # DₖRₖ = V̅ₖ with v̅ₖ = cpₖvₖ + spₖqₖ₋₁
      if iter == 1
        @kaxpy!(n, one(T)/ρ, cpₖ * v, d)
      else
        @kaxpby!(n, one(T)/ρ, cpₖ * v, -θ/ρ, d)
        @kaxpy!(n, one(T)/ρ, spₖ * q, d)
        @kaxpby!(n, spₖ, v, -cpₖ, q)  # q̄ₖ ← spₖ * vₖ - cpₖ * qₖ₋₁
      end
    else
      # DₖRₖ = Vₖ
      if iter == 1
        @kaxpy!(n, one(T)/ρ, v, d)
      else
        @kaxpby!(n, one(T)/ρ, v, -θ/ρ, d)
      end
    end

    # xₖ = Dₖzₖ
    @kaxpy!(n, ζ, d, x)

    # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    mul!(Aᵀu, Aᵀ, u)
    @kaxpby!(n, one(T), Aᵀu, -β, Nv)
    NisI || mul!(v, N, Nv)
    α = sqrt(@kdot(n, v, Nv))
    Anorm² = Anorm² + α * α  # = ‖Lₖ‖
    ArNorm = α * β * abs(ζ/ρ)
    history && push!(ArNorms, ArNorm)

    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", 1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm²)

    if λ > 0
      (cdₖ, sdₖ, λₖ₊₁) = sym_givens(λ, λₐᵤₓ)
      @kscal!(n, sdₖ, q)  # qₖ ← sdₖ * q̄ₖ
      (cpₖ, spₖ, αhat) = sym_givens(α, λₖ₊₁)
    else
      αhat = α
    end

    if α ≠ 0
      @kscal!(n, one(T)/α, v)
      NisI || @kscal!(n, one(T)/α, Nv)
      @kaxpby!(m, one(T) / αhat, u, -βhat / αhat, wbar)  # wbar = (u - beta * wbar) / alpha
    end
    θ    =  s * αhat
    ρbar = -c * αhat

    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) & (ArNorm ≤ ɛ_i)
    tired  = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (solved ? "found approximate minimum-norm solution" : "found approximate minimum least-squares solution")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, y, stats)
end
