# An implementation of BICGSTAB for the solution of unsymmetric and square consistent linear system Ax = b.
#
# This method is described in
#
# H. A. van der Vorst
# Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems.
# SIAM Journal on Scientific and Statistical Computing, 13(2), pp. 631--644, 1992.
#
# G. L.G. Sleijpen and D. R. Fokkema
# BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum.
# Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montréal, October 2020.

export bicgstab, bicgstab!

"""
    (x, stats) = bicgstab(A, b::AbstractVector{FC}; c::AbstractVector{FC}=b,
                          M=I, N=I, atol::T=√eps(T), rtol::T=√eps(T),
                          itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the square linear system Ax = b using the BICGSTAB method.
BICGSTAB requires two initial vectors `b` and `c`.
The relation `bᵀc ≠ 0` must be satisfied and by default `c = b`.

The Biconjugate Gradient Stabilized method is a variant of BiCG, like CGS,
but using different updates for the Aᵀ-sequence in order to obtain smoother
convergence than CGS.

If BICGSTAB stagnates, we recommend DQGMRES and BiLQ as alternative methods for unsymmetric square systems.

BICGSTAB stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖b‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if verbose mode is enabled (verbose > 0).
Information will be displayed every `verbose` iterations.

This implementation allows a left preconditioner `M` and a right preconditioner `N`.

#### References

* H. A. van der Vorst, [*Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems*](https://doi.org/10.1137/0913035), SIAM Journal on Scientific and Statistical Computing, 13(2), pp. 631--644, 1992.
* G. L.G. Sleijpen and D. R. Fokkema, *BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum*, Electronic Transactions on Numerical Analysis, 1, pp. 11--32, 1993.
"""
function bicgstab(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = BicgstabSolver(A, b)
  bicgstab!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

function bicgstab!(solver :: BicgstabSolver{T,FC,S}, A, b :: AbstractVector{FC}; c :: AbstractVector{FC}=b,
                   M=I, N=I, atol :: T=√eps(T), rtol :: T=√eps(T),
                   itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  (verbose > 0) && @printf("BICGSTAB: system of size %d\n", n)

  # Check M == Iₙ and N == Iₙ
  MisI = (M == I)
  NisI = (N == I)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")
  MisI || (promote_type(eltype(M), T) == T) || error("eltype(M) can't be promoted to $T")
  NisI || (promote_type(eltype(N), T) == T) || error("eltype(N) can't be promoted to $T")

  # Set up workspace.
  allocate_if(!MisI, solver, :t , S, n)
  allocate_if(!NisI, solver, :yz, S, n)
  x, r, p, v, s, qd, stats = solver.x, solver.r, solver.p, solver.v, solver.s, solver.qd, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  q = d = solver.qd
  t = MisI ? d : solver.t
  y = NisI ? p : solver.yz
  z = NisI ? s : solver.yz

  x .= zero(T)   # x₀
  s .= zero(T)   # s₀
  v .= zero(T)   # v₀
  mul!(r, M, b)  # r₀
  p .= r         # p₁

  α = one(T) # α₀
  ω = one(T) # ω₀
  ρ = one(T) # ρ₀

  # Compute residual norm ‖r₀‖₂.
  rNorm = @knrm2(n, r)
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.solved, stats.inconsistent = true, false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = 2*n)

  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s  %8s  %8s\n", "k", "‖rₖ‖", "αₖ", "ωₖ")
  display(iter, verbose) && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, α, ω)

  next_ρ = @kdot(n, r, c)  # ρ₁ = ⟨r₀,r̅₀⟩
  if next_ρ == 0
    stats.solved, stats.inconsistent = false, false
    stats.status = "Breakdown bᵀc = 0"
    return solver
  end

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"

  while !(solved || tired || breakdown)
    # Update iteration index and ρ.
    iter = iter + 1
    ρ = next_ρ

    NisI || mul!(y, N, p)                # yₖ = N⁻¹pₖ
    mul!(q, A, y)                        # qₖ = Ayₖ
    mul!(v, M, q)                        # vₖ = M⁻¹qₖ
    α = ρ / @kdot(n, v, c)               # αₖ = ⟨rₖ₋₁,r̅₀⟩ / ⟨vₖ,r̅₀⟩
    @kcopy!(n, r, s)                     # sₖ = rₖ₋₁
    @kaxpy!(n, -α, v, s)                 # sₖ = sₖ - αₖvₖ
    @kaxpy!(n, α, y, x)                  # xₐᵤₓ = xₖ₋₁ + αₖyₖ
    NisI || mul!(z, N, s)                # zₖ = N⁻¹sₖ
    mul!(d, A, z)                        # dₖ = Azₖ
    MisI || mul!(t, M, d)                # tₖ = M⁻¹dₖ
    ω = @kdot(n, t, s) / @kdot(n, t, t)  # ⟨tₖ,sₖ⟩ / ⟨tₖ,tₖ⟩
    @kaxpy!(n, ω, z, x)                  # xₖ = xₐᵤₓ + ωₖzₖ
    @kcopy!(n, s, r)                     # rₖ = sₖ
    @kaxpy!(n, -ω, t, r)                 # rₖ = rₖ - ωₖtₖ
    next_ρ = @kdot(n, r, c)              # ρₖ₊₁ = ⟨rₖ,r̅₀⟩
    β = (next_ρ / ρ) * (α / ω)           # βₖ₊₁ = (ρₖ₊₁ / ρₖ) * (αₖ / ωₖ)
    @kaxpy!(n, -ω, v, p)                 # pₐᵤₓ = pₖ - ωₖvₖ
    @kaxpby!(n, one(T), r, β, p)         # pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₐᵤₓ

    # Compute residual norm ‖rₖ‖₂.
    rNorm = @knrm2(n, r)
    history && push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = (α == 0 || isnan(α))
    display(iter, verbose) && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, α, ω)
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (breakdown ? "breakdown αₖ == 0" : "solution good enough given atol and rtol")

  # Update stats
  stats.solved = solved
  stats.inconsistent = false
  stats.status = status
  return solver
end
