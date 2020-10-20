# An implementation of DIOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Iterative methods for sparse linear systems.
# PWS Publishing Company, Boston, USA, 1996.
#
# Y. Saad, Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems.
# SIAM journal on scientific and statistical computing, 5(1), pp. 203--228, 1984.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2018.

export diom

"""
    (x, stats) = diom(A, b; M, N, atol, rtol, itmax, memory, pivoting, verbose)

Solve the consistent linear system Ax = b using direct incomplete orthogonalization method.

DIOM is similar to CG with partial reorthogonalization.

An advantage of DIOM is that nonsymmetric or symmetric indefinite or both nonsymmetric
and indefinite systems of linear equations can be handled by this single algorithm.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u
"""
function diom(A, b :: AbstractVector{T};
              M=opEye(), N=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
              memory :: Int=20, pivoting :: Bool=false, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("DIOM: system of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")
  isa(N, opEye) || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial solution x₀ and residual r₀.
  x = kzeros(S, n)  # x₀
  x_old = copy(x)
  r₀ = M * b      # M⁻¹(b - Ax₀)
  # Compute β.
  rNorm = @knrm2(n, r₀) # β = ‖r₀‖₂
  rNorm == 0 && return x, SimpleStats(true, false, [rNorm], T[], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  mem = min(memory, itmax) # Memory.
  V = [kzeros(S, n) for i = 1 : mem]  # Preconditioned Krylov vectors, orthogonal basis for {b, M⁻¹AN⁻¹b, (M⁻¹AN⁻¹)²b, ..., (M⁻¹AN⁻¹)ᵐ⁻¹b}.
  P = [kzeros(S, n) for i = 1 : mem]  # Directions for x : Pₘ = Vₘ(Uₘ)⁻¹.
  H = zeros(T, mem+2)                 # Last column of the band hessenberg matrix Hₘ = LₘUₘ.
  # Each column has at most mem + 1 nonzero elements. hᵢ.ₘ is stored as H[m-i+2].
  # m-i+2 represents the indice of the diagonal where hᵢ.ₘ is located.
  # In addition of that, the last column of Uₘ is stored in H.
  L = zeros(T, mem)        # Last mem Pivots of Lₘ.
  p = BitArray(undef, mem) # Last mem permutations.

  # Initial ξ₁ and V₁.
  ξ = rNorm
  @. V[1] = r₀ / rNorm

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index.
    iter = iter + 1

    # Set position in circulars stacks.
    pos = mod(iter-1, mem) + 1 # Position corresponding to pₘ and vₘ in circular stacks P and V.
    next_pos = mod(iter, mem) + 1 # Position corresponding to vₘ₊₁ in the circular stack V.

    # Incomplete Arnoldi procedure.
    z = N * V[pos] # N⁻¹vₘ, forms pₘ
    t = A * z      # AN⁻¹vₘ
    w = M * t      # M⁻¹AN⁻¹vₘ, forms vₘ₊₁
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem) + 1 # Position corresponding to vᵢ in the circular stack V.
      diag = iter - i + 2
      H[diag] = @kdot(n, w, V[ipos]) # hᵢ.ₘ = < M⁻¹AN⁻¹vₘ , vᵢ >
      @kaxpy!(n, -H[diag], V[ipos], w) # w ← w - hᵢ.ₘ * vᵢ
    end
    # Compute hₘ₊₁.ₘ and vₘ₊₁.
    H[1] = @knrm2(n, w) # hₘ₊₁.ₘ = ‖vₘ₊₁‖₂
    if H[1] ≠ 0 # hₘ₊₁.ₘ = 0 ⇒ "lucky breakdown"
      @. V[next_pos] = w / H[1] # vₘ₊₁ = w / hₘ₊₁.ₘ
    end
    # It's possible that uₘ₋ₘₑₘ.ₘ ≠ 0 when m ≥ mem + 1
    if iter ≥ mem + 2
      H[mem+2] = zero(T) # hₘ₋ₘₑₘ.ₘ = 0
    end

    # Update the LU factorization with partial pivoting of H.
    # Compute the last column of Uₘ.
    if iter ≥ 2
      for i = max(2,iter-mem+1) : iter
        lpos = mod(i-1, mem) + 1 # Position corresponding to lᵢ.ᵢ₋₁ in the circular stack L.
        diag = iter - i + 2
        next_diag = diag + 1
        if p[lpos]
          # The rows i-1 and i are permuted.
          H[diag], H[next_diag] = H[next_diag], H[diag]
        end
        # uᵢ.ₘ ← hᵢ.ₘ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₘ
        H[diag] = H[diag] - L[lpos] * H[next_diag]
      end
      # Compute ξₘ the last component of zₘ = β(Lₘ)⁻¹e₁.
      if !p[pos] # p[pos] ⇒ ξₘ = ξₘ₋₁
        # ξₘ = -lₘ.ₘ₋₁ * ξₘ₋₁
        ξ = - L[pos] * ξ
      end
    end

    # Compute the direction pₘ, the last column of Pₘ = Vₘ(Uₘ)⁻¹.
    for i = max(1,iter-mem) : iter-1
      ipos = mod(i-1, mem) + 1 # Position corresponding to pᵢ in the circular stack P.
      diag = iter - i + 2
      if ipos == pos
        # pₐᵤₓ ← -hₘ₋ₘₑₘ.ₘ * pₘ₋ₘₑₘ
        @kscal!(n, -H[diag], P[pos])
      else
        # pₐᵤₓ ← pₐᵤₓ - hᵢ.ₘ * pᵢ
        @kaxpy!(n, -H[diag], P[ipos], P[pos])
      end
    end
    # pₐᵤₓ ← pₐᵤₓ + N⁻¹vₘ
    @kaxpy!(n, one(T), z, P[pos])

    # Determine if interchange between hₘ₊₁.ₘ and uₘ.ₘ is needed and compute next pivot lₘ₊₁.ₘ.
    if pivoting && abs(H[2]) < H[1]
      p[next_pos] = true
      # pₘ = pₐᵤₓ / hₘ₊₁.ₘ
      @. P[pos] = P[pos] / H[1]
      # lₘ₊₁.ₘ = uₘ.ₘ / hₘ₊₁.ₘ
      L[next_pos] = H[2] / H[1]
    else
      p[next_pos] = false
      # pₘ = pₐᵤₓ / uₘ.ₘ
      @. P[pos] = P[pos] / H[2]
      # lₘ₊₁.ₘ = hₘ₊₁.ₘ / uₘ.ₘ
      L[next_pos] = H[1] / H[2]
    end

    # Compute solution xₘ.
    if p[pos]
      # xₘ = xₘ₋ₙ + ξₘ₋ₙ * pₘ
      # x_old = xₘ₋ₙ, with m-n is the last iteration without permutation at the next step
      @. x = x_old + ξ * P[pos]
    else
      # xₘ = xₘ₋₁ + ξₘ * pₘ
      @kaxpy!(n, ξ, P[pos], x)
    end

    # Update x_old and residual norm.
    if !p[next_pos]
      @. x_old = x
      # ‖ M⁻¹(b - Axₘ) ‖₂ = hₘ₊₁.ₘ * |ξₘ / uₘ.ₘ| without pivoting
      rNorm = H[1] * abs(ξ / H[2])
    else
      # ‖ M⁻¹(b - Axₘ) ‖₂ = hₘ₊₁.ₘ * |ξₘ / hₘ₊₁.ₘ| = |ξₘ| with pivoting
      rNorm = abs(ξ)
    end
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
