# An implementation of FOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Krylov subspace methods for solving unsymmetric linear systems.
# Mathematics of computation, Vol. 37(155), pp. 105--126, 1981.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, December 2018.

export fom

"""Solve the consistent linear system Ax = b using FOM method.

FOM algorithm is based on the Arnoldi orthogonalization process.

When x₀ ≠ 0 (restarted version), FOM resolves A * Δx = r₀ with r₀ = b - Ax₀ and returns x = x₀ + Δx.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u
"""
function fom(A :: AbstractLinearOperator, b :: AbstractVector{T};
             x0 :: AbstractVector{T}=zeros(T, size(A,1)),
             M :: AbstractLinearOperator=opEye(),
             N :: AbstractLinearOperator=opEye(),
             atol :: T=√eps(T), rtol :: T=√eps(T),
             pivoting :: Bool=false, reorthogonalization :: Bool=false,
             itmax :: Int=0, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("FOM: system of size %d\n", n)

  # Initial solution x₀ and residual r₀.
  x = x0 # x₀
  r₀ = A * x0
  @. r₀ = -r₀ + b
  r₀ = M * r₀ # r₀ = M⁻¹(b - Ax₀)

  # Compute β
  rNorm = @knrm2(n, r₀) # β = ‖r₀‖₂
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  V  = Vector{T}[zeros(T, n)] # Preconditioned Krylov vectors, orthogonal basis for {r₀, M⁻¹AN⁻¹r₀, (M⁻¹AN⁻¹)²r₀, ..., (M⁻¹AN⁻¹)ᵐ⁻¹r₀}.
  l  = T[]         # Coefficients used for the factorization LₖUₖ = Hₖ.
  H  = Vector{T}[] # Hessenberg matrix Hₖ.
  z  = T[]         # Right-hand of the least squares problem Hₖyₖ = βe₁ ⟺ Uₖyₖ = zₖ with Lₖzₖ = βe₁.
  p = BitArray(undef, 0) # Row permutations.

  # Initial ζ₁ and V₁.
  push!(z, rNorm)
  @. V[1] = r₀ / rNorm

  # Stopping criterion
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index
    iter = iter + 1

    # Update workspace
    push!(H, zeros(T, iter))
    push!(l, zero(T))
    push!(p, false)

    # Arnoldi procedure
    NV = N * V[iter] # N⁻¹vₖ
    AV = A * NV      # AN⁻¹vₖ
    MV = M * AV      # M⁻¹AN⁻¹vₖ
    for i = 1 : iter
      H[iter][i] = @kdot(n, V[i], MV)
      @kaxpy!(n, -H[iter][i], V[i], MV)
    end

    # Reorthogonalization
    if reorthogonalization
      for i = 1 : iter
        Htmp =  @kdot(n, V[i], MV)
        H[iter][i] += Htmp
        @kaxpy!(n, -Htmp, V[i], MV)
      end
    end

    # Compute hₖ₊₁.ₖ
    Hbis = @knrm2(n, MV) # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

    # Update the LU factorization of H.
    if iter ≥ 2
      push!(z, zero(T))
      for i = 2 : iter
        if p[i-1]
          # The rows i-1 and i are permuted.
          H[iter][i-1], H[iter][i] = H[iter][i], H[iter][i-1]
        end
        # uᵢ.ₖ ← hᵢ.ₖ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₖ
        H[iter][i] = H[iter][i] - l[i-1] * H[iter][i-1]
      end
      if p[iter-1]
        # ζₖ = ζₖ₋₁
        z[iter] = z[iter-1]
        # ζₖ₋₁ = 0
        z[iter-1] = zero(T)
      else
        # ζₖ = -lₖ.ₖ₋₁ * ζₖ₋₁
        z[iter] = - l[iter-1] * z[iter-1]
      end
    end

    # Determine if interchange between hₖ₊₁.ₖ and uₖ.ₖ is needed and compute next pivot lₖ₊₁.ₖ.
    if pivoting && abs(H[iter][iter]) < Hbis
      p[iter] = true
      # lₖ₊₁.ₖ = uₖ.ₖ / hₖ₊₁.ₖ
      l[iter] = H[iter][iter] / Hbis
      # uₖ.ₖ ← hₖ₊₁.ₖ
      H[iter][iter] = Hbis
      # ‖ M⁻¹(b - Axₖ) ‖₂ = hₖ₊₁.ₖ * |ζₖ / hₖ₊₁.ₖ| = |ζₖ| with pivoting
      rNorm = abs(z[iter])
    else
      p[iter] = false
      # lₖ₊₁.ₖ = hₖ₊₁.ₖ / uₖ.ₖ
      l[iter] = Hbis / H[iter][iter]
      # ‖ M⁻¹(b - Axₖ) ‖₂ = hₖ₊₁.ₖ * |ζₖ / uₖ.ₖ| without pivoting
      rNorm = Hbis * abs(z[iter] / H[iter][iter])
    end

    # Update residual norm estimate.
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε || Hbis == zero(T) # hₖ₊₁.ₖ = 0 ⇒ "lucky breakdown"
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)

    # Compute vₖ₊₁.
    if !(solved || tired)
       push!(V, zeros(T, n))
       @. V[iter+1] = MV / Hbis # Normalization of vₖ₊₁
    end

  end
  verbose && @printf("\n")

  # Compute yₖ by solving Uₖyₖ = zₖ with backward substitution
  # We will overwrite zₖ with yₖ
  y = z
  y[iter] = z[iter] / H[iter][iter]
  for i = iter-1 : -1 : 1
    y[i] = (z[i] - sum(H[k][i] * y[k] for k = i+1 : iter)) / H[i][i]
  end

  # Form xₖ = x₀ + Vₖyₖ
  for i = 1 : iter
    Dᵢ = N * V[i] # Needed with right preconditioning
    @kaxpy!(n, y[i], Dᵢ, x)
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
