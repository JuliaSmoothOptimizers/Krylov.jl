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

export bicgstab

"""
    (x, stats) = bicgstab(A, b; c, M, N, atol, rtol, itmax, verbose)

Solve the square linear system Ax = b using the BICGSTAB method.

The Biconjugate Gradient Stabilized method is a variant of BiCG, like CGS,
but using different updates for the Aᵀ-sequence in order to obtain smoother
convergence than CGS.

If BICGSTAB stagnates, we recommend DQGMRES and BiLQ as alternative methods for unsymmetric square systems.

BICGSTAB stops when `itmax` iterations are reached or when `‖rₖ‖ ≤ atol + ‖b‖ * rtol`.
`atol` is an absolute tolerance and `rtol` is a relative tolerance.

Additional details can be displayed if the `verbose` mode is enabled.

This implementation allows a left preconditioner `M` and a right preconditioner `N`.
"""
function bicgstab(A, b :: AbstractVector{T}; c :: AbstractVector{T}=b,
                   M=opEye(), N=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T),
                   itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("BICGSTAB: system of size %d\n", n)

  # Check M == Iₘ and N == Iₙ
  MisI = isa(M, opEye)
  NisI = isa(N, opEye)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  MisI || (eltype(M) == T) || error("eltype(M) ≠ $T")
  NisI || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Set up workspace.
  x = kzeros(S, n)  # x₀
  s = kzeros(S, n)  # s₀
  v = kzeros(S, n)  # v₀
  r = copy(b)       # r₀
  p = copy(r)       # p₁

  α = one(T) # α₀
  ω = one(T) # ω₀
  ρ = one(T) # ρ₀

  # Initial residual norm ‖r₀‖.
  rNorm = @knrm2(n, b)
  rNorm == 0 && return (x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution"))

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5s  %7s  %8s  %8s\n", "k", "‖rₖ‖", "αₖ", "ωₖ")
  verbose && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, α, ω)

  next_ρ = @kdot(n, b, c)  # ρ₁ = ⟨r₀,r̅₀⟩ = ⟨b,c⟩
  next_ρ == 0 && return (x, SimpleStats(false, false, [bNorm], T[], "Breakdown bᵀc = 0"))

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  breakdown = false
  status = "unknown"

  while !(solved || tired || breakdown)
    # Update iteration index and ρ.
    iter = iter + 1
    ρ = next_ρ 

    y = N * p                                # yₖ = Npₖ
    v .= A * y                               # vₖ = Ayₖ
    α = ρ / @kdot(n, v, c)                   # αₖ = ⟨rₖ₋₁,r̅₀⟩ / ⟨vₖ,r̅₀⟩
    @. s = r - α * v                         # sₖ = rₖ₋₁ - αₖvₖ
    @kaxpy!(n, α, y, x)                      # xₐᵤₓ = xₖ₋₁ + αₖyₖ
    z = N * s                                # zₖ = Nsₖ
    t = A * z                                # tₖ = Azₖ
    MisI ? (Ms = s) : (r .= M * s ; Ms = r)  # Msₖ
    Mt = M * t                               # Mtₖ
    ω = @kdot(n, Mt, Ms) / @kdot(n, Mt, Mt)  # ⟨Mtₖ,Msₖ⟩ / ⟨Mtₖ,Mtₖ⟩
    @kaxpy!(n, ω, z, x)                      # xₖ = xₐᵤₓ + ωₖzₖ
    @. r = s - ω * t                         # rₖ = sₖ - ωₖtₖ
    next_ρ = @kdot(n, r, c)                  # ρₖ₊₁ = ⟨rₖ,r̅₀⟩
    β = (next_ρ / ρ) * (α / ω)               # βₖ₊₁ = (ρₖ₊₁ / ρₖ) * (αₖ / ωₖ)
    @kaxpy!(n, -ω, v, p)                     # pₐᵤₓ = pₖ - ωₖvₖ
    @kaxpby!(n, one(T), r, β, p)             # pₖ₊₁ = rₖ₊₁ + βₖ₊₁pₐᵤₓ

    # Compute residual norm ‖rₖ‖₂.
    rNorm = @knrm2(n, r)
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = (α == 0 || isnan(α))
    verbose && @printf("%5d  %7.1e  %8.1e  %8.1e\n", iter, rNorm, α, ω)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (breakdown ? "breakdown αₖ == 0" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
