# An implementation of CRAIG-MR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
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
                            M=I, N=I, λ::T=zero(T), atol::T=√eps(T),
                            rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the consistent linear system

    Ax + √λs = b

using the CRAIG-MR method, where λ ≥ 0 is a regularization parameter.
This method is equivalent to applying the Conjugate Residuals method
to the normal equations of the second kind

    (AAᵀ + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

    min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

    min ‖(x,s)‖₂  s.t. Ax + √λs = b.

Preconditioners M⁻¹ and N⁻¹ may be provided in the form of linear operators and are
assumed to be symmetric and positive definite.
Afterward CRAIGMR solves the symmetric and quasi-definite system

    [ -N   Aᵀ ] [ x ]   [ 0 ]
    [  A   M  ] [ y ] = [ b ],

which is equivalent to applying MINRES to (M + AN⁻¹Aᵀ)y = b.

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
                  M=I, N=I, λ :: T=zero(T), atol :: T=√eps(T),
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

  # Set up workspace.
  allocate_if(!MisI, solver, :u, S, m)
  allocate_if(!NisI, solver, :v, S, n)
  x, Nv, Aᵀu, y, Mu, w, wbar, Av = solver.x, solver.Nv, solver.Aᵀu, solver.y, solver.Mu, solver.w, solver.wbar, solver.Av
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

  # Initialize other constants.
  ζbar = β
  ρbar = α
  θ = zero(T)
  rNorm = ζbar
  rNorms = history ? [rNorm] : T[]
  ArNorm = α
  ArNorms = history ? [ArNorm] : T[]

  ɛ_c = atol + rtol * rNorm  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm  # Stopping tolerance for inconsistent systems.

  wbar .= u
  @kscal!(m, one(T)/α, wbar)
  w .= zero(T)

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
    (c, s, ρ) = sym_givens(ρbar, β)
    ζ = c * ζbar
    ζbar = s * ζbar
    rNorm = abs(ζbar)
    history && push!(rNorms, rNorm)

    @kaxpby!(m, one(T)/ρ, wbar, -θ/ρ, w)  # w = (wbar - θ * w) / ρ
    @kaxpy!(m, ζ, w, y)             # y = y + ζ * w

    # 2. αₖ₊₁Nvₖ₊₁ = Aᵀuₖ₊₁ - βₖ₊₁Nvₖ
    mul!(Aᵀu, Aᵀ, u)
    @kaxpby!(n, one(T), Aᵀu, -β, Nv)
    NisI || mul!(v, N, Nv)
    α = sqrt(@kdot(n, v, Nv))
    Anorm² = Anorm² + α * α  # = ‖Lₖ‖
    ArNorm = α * β * abs(ζ/ρ)
    history && push!(ArNorms, ArNorm)

    display(iter, verbose) && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n", 1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm²)

    if α ≠ 0
      @kscal!(n, one(T)/α, v)
      NisI || @kscal!(n, one(T)/α, Nv)
      @kaxpby!(m, one(T)/α, u, -β/α, wbar)  # wbar = (u - beta * wbar) / alpha
    end
    θ = s * α
    ρbar = -c * α

    solved = rNorm ≤ ɛ_c
    inconsistent = (rNorm > 100 * ɛ_c) & (ArNorm ≤ ɛ_i)
    tired  = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  Aᵀy = Aᵀu
  mul!(Aᵀy, Aᵀ, y)
  N⁻¹Aᵀy = NisI ? Aᵀy : solver.v
  mul!(N⁻¹Aᵀy, N, Aᵀy)
  @. x = N⁻¹Aᵀy

  status = tired ? "maximum number of iterations exceeded" : (solved ? "found approximate minimum-norm solution" : "found approximate minimum least-squares solution")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status)
  return (x, y, stats)
end
