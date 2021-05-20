# An implementation of CGS for the solution of the square linear system Ax = b.
#
# This method is described in
#
# P. Sonneveld, CGS, A Fast Lanczos-Type Solver for Nonsymmetric Linear systems.
# SIAM Journal on Scientific and Statistical Computing, 10(1), pp. 36--52, 1989.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, October 2018.

export cgs, cgs!

"""
    (x, stats) = cgs(A, b::AbstractVector{T}; c::AbstractVector{T}=b,
                     M=opEye(), N=opEye(), atol::T=√eps(T), rtol::T=√eps(T),
                     itmax::Int=0, verbose::Int=0, history::Bool=false) where T <: AbstractFloat

Solve the consistent linear system Ax = b using conjugate gradient squared algorithm.

From "Iterative Methods for Sparse Linear Systems (Y. Saad)" :

«The method is based on a polynomial variant of the conjugate gradients algorithm.
Although related to the so-called bi-conjugate gradients (BCG) algorithm,
it does not involve adjoint matrix-vector multiplications, and the expected convergence
rate is about twice that of the BCG algorithm.

The Conjugate Gradient Squared algorithm works quite well in many cases.
However, one difficulty is that, since the polynomials are squared, rounding errors
tend to be more damaging than in the standard BCG algorithm. In particular, very
high variations of the residual vectors often cause the residual norms computed
to become inaccurate.

TFQMR and BICGSTAB were developed to remedy this difficulty.»

This implementation allows a left preconditioner M and a right preconditioner N.

#### Reference

* P. Sonneveld, *CGS, A Fast Lanczos-Type Solver for Nonsymmetric Linear systems*, SIAM Journal on Scientific and Statistical Computing, 10(1), pp. 36--52, 1989.
"""
function cgs(A, b :: AbstractVector{T}; kwargs...) where T <: AbstractFloat
  solver = CgsSolver(A, b)
  cgs!(solver, A, b; kwargs...)
end

function cgs!(solver :: CgsSolver{T,S}, A, b :: AbstractVector{T}; c :: AbstractVector{T}=b,
              M=opEye(), N=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T),
              itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, S <: DenseVector{T}}

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("CGS: system of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")
  isa(N, opEye) || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Set up workspace.
  x, r, u, p, q = solver.x, solver.r, solver.u, solver.p, solver.q

  x .= zero(T)  # x₀
  r .= M * b    # r₀

  # Compute residual norm ‖r₀‖₂.
  rNorm = @knrm2(n, r)
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  # Compute ρ₀ = ⟨ r₀,̅r₀ ⟩
  ρ = @kdot(n, r, c)
  ρ == 0 && return x, SimpleStats(false, false, [rNorm], T[], "Breakdown bᵀc = 0")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = history ? [rNorm] : T[]
  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  u .= r        # u₀
  p .= r        # p₀
  q .= zero(T)  # q₋₁

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"

  while !(solved || tired || breakdown)

    mul!(y, N, p)                 # yₖ = N⁻¹pₖ
    mul!(t, A, y)                 # tₖ = Ayₖ
    mul!(v, M, t)                 # vₖ = M⁻¹tₖ
    σ = @kdot(n, v, c)            # σₖ = ⟨ M⁻¹AN⁻¹pₖ,̅r₀ ⟩
    α = ρ / σ                     # αₖ = ρₖ / σₖ
    @kcopy!(n, u, q)              # qₖ = uₖ
    @kaxpy!(n, -α, v, q)          # qₖ = qₖ - αₖ * M⁻¹AN⁻¹pₖ
    @kaxpy!(n, one(T), q, u)      # uₖ₊½ = uₖ + qₖ
    mul!(z, N, u)                 # zₖ = N⁻¹uₖ₊½
    @kaxpy!(n, α, z, x)           # xₖ₊₁ = xₖ + αₖ * N⁻¹(uₖ + qₖ)
    mul!(s, A, z)                 # sₖ = Azₖ
    mul!(w, M, s)                 # wₖ = M⁻¹sₖ
    @kaxpy!(n, -α, w, r)          # rₖ₊₁ = rₖ - αₖ * M⁻¹AN⁻¹(uₖ + qₖ)
    ρ_next = @kdot(n, r, c)       # ρₖ₊₁ = ⟨ rₖ₊₁,̅r₀ ⟩
    β = ρ_next / ρ                # βₖ = ρₖ₊₁ / ρₖ
    @kcopy!(n, r, u)              # uₖ₊₁ = rₖ₊₁
    @kaxpy!(n, β, q, u)           # uₖ₊₁ = uₖ₊₁ + βₖ * qₖ
    @kaxpby!(n, one(T), q, β, p)  # pₐᵤₓ = qₖ + βₖ * pₖ
    @kaxpby!(n, one(T), u, β, p)  # pₖ₊₁ = uₖ₊₁ + βₖ * pₐᵤₓ

    # Update ρ.
    ρ = ρ_next # ρₖ ← ρₖ₊₁

    # Update iteration index.
    iter = iter + 1

    # Compute residual norm ‖rₖ‖₂.
    rNorm = @knrm2(n, r)
    history && push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = (α == 0 || isnan(α))
    display(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (breakdown ? "breakdown αₖ == 0" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
