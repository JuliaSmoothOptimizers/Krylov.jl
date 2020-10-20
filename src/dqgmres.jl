# An implementation of DQGMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Iterative methods for sparse linear systems.
# PWS Publishing Company, Boston, USA, 1996.
#
# Y. Saad and K. Wu, DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization.
# Numerical Linear Algebra with Applications, Vol. 3(4), pp. 329--343, 1996.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, August 2018.

export dqgmres

"""
    (x, stats) = dqgmres(A, b; M, N, atol, rtol, itmax, memory, verbose)

Solve the consistent linear system Ax = b using DQGMRES method.

DQGMRES algorithm is based on the incomplete Arnoldi orthogonalization process
and computes a sequence of approximate solutions with the quasi-minimal residual property.

This implementation allows a left preconditioner M and a right preconditioner N.
- Left  preconditioning : M⁻¹Ax = M⁻¹b
- Right preconditioning : AN⁻¹u = b with x = N⁻¹u
- Split preconditioning : M⁻¹AN⁻¹u = M⁻¹b with x = N⁻¹u
"""
function dqgmres(A, b :: AbstractVector{T};
                 M=opEye(), N=opEye(), atol :: T=√eps(T), rtol :: T=√eps(T),
                 itmax :: Int=0, memory :: Int=20, verbose :: Bool=false) where T <: AbstractFloat

  m, n = size(A)
  m == n || error("System must be square")
  length(b) == m || error("Inconsistent problem size")
  verbose && @printf("DQGMRES: system of size %d\n", n)

  # Check type consistency
  eltype(A) == T || error("eltype(A) ≠ $T")
  isa(M, opEye) || (eltype(M) == T) || error("eltype(M) ≠ $T")
  isa(N, opEye) || (eltype(N) == T) || error("eltype(N) ≠ $T")

  # Determine the storage type of b
  S = typeof(b)

  # Initial solution x₀ and residual r₀.
  x = kzeros(S, n)  # x₀
  r₀ = M * b        # M⁻¹(b - Ax₀)
  # Compute β
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
  P = [kzeros(S, n) for i = 1 : mem]  # Directions for x : Pₘ = Vₘ(Rₘ)⁻¹.
  s = zeros(T, mem)                   # Last mem Givens sines used for the factorization QₘRₘ = Hₘ.
  c = zeros(T, mem)                   # Last mem Givens cosines used for the factorization QₘRₘ = Hₘ.
  H = zeros(T, mem+2)                 # Last column of the band hessenberg matrix Hₘ.
  # Each column has at most mem + 1 nonzero elements. hᵢ.ₘ is stored as H[m-i+2].
  # m-i+2 represents the indice of the diagonal where hᵢ.ₘ is located.
  # In addition of that, the last column of Rₘ is also stored in H.

  # Initial γ₁ and V₁.
  γₘ = rNorm # γₘ and γₘ₊₁ are the last components of gₘ, right-hand of the least squares problem min ‖ Hₘyₘ - gₘ ‖₂.
  @. V[1] = r₀ / rNorm

  # The following stopping criterion compensates for the lag in the
  # residual, but usually increases the number of iterations.
  # solved = sqrt(max(1, iter-mem+1)) * |γₘ₊₁| ≤ ε
  solved = rNorm ≤ ε # less accurate, but acceptable.
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
    # rₘ₋ₘₑₘ.ₘ ≠ 0 when m ≥ mem + 1
    if iter ≥ mem + 2
      H[mem+2] = zero(T) # hₘ₋ₘₑₘ.ₘ = 0
    end

    # Update the QR factorization of H.
    # Apply mem previous Givens reflections Ωᵢ.
    for i = max(1,iter-mem) : iter-1
      irot_pos = mod(i-1, mem) + 1 # Position corresponding to cᵢ and sᵢ in circular stacks c and s.
      diag = iter - i + 1
      next_diag = diag + 1
      H_aux        = c[irot_pos] * H[next_diag] + s[irot_pos] * H[diag]
      H[diag]      = s[irot_pos] * H[next_diag] - c[irot_pos] * H[diag]
      H[next_diag] = H_aux
    end

    # Compute and apply current Givens reflection Ωₘ.
    # [cₘ  sₘ] [ hₘ.ₘ ] = [ρₘ]
    # [sₘ -cₘ] [hₘ₊₁.ₘ]   [0 ]
    (c[pos], s[pos], H[2]) = sym_givens(H[2], H[1])
    γₘ₊₁ = s[pos] * γₘ
    γₘ   = c[pos] * γₘ

    # Compute the direction pₘ, the last column of Pₘ = Vₘ(Rₘ)⁻¹.
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
    # pₘ = pₐᵤₓ / hₘ.ₘ
    @. P[pos] = P[pos] / H[2]

    # Compute solution xₘ.
    # xₘ ← xₘ₋₁ + γₘ * pₘ
    @kaxpy!(n, γₘ, P[pos], x)

    # Update residual norm estimate.
    # ‖ M⁻¹(b - Axₘ) ‖₂ ≈ |γₘ₊₁|
    rNorm = abs(γₘ₊₁)
    push!(rNorms, rNorm)

    # Update γₘ.
    γₘ = γₘ₊₁

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
