# An implementation of DIOM for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Iterative methods for sparse linear systems.
# PWS Publishing Company, Boston, USA, 1996.
#
# Y. Saad, Practical use of some krylov subspace methods for solving indefinite and nonsymmetric linear systems.
# SIAM journal on scientific and statistical computing.
# Vol. 5, No. 1, March 1984.
# 
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, September 2018.

export diom

"""Solve the consistent linear system Ax = b using direct incomplete orthogonalization method.

DIOM is similar to SYMMLQ for nonsymmetric problems.
It's a more economical algorithm based upon the LU factorization with partial pivoting.

In the particular case where A is symmetric indefinite, DIOM is theorecally equivalent 
to SYMMLQ but slightly more economical. 

An advantage of DIOM is that nonsymmetric or symmetric indefinite or both nonsymmetric 
and indefinite systems of linear equations can be handled by this single algorithm.

This implementation allows a right preconditioning with M.
"""
function diom{T <: Number}(A :: AbstractLinearOperator, b :: AbstractVector{T};
                              M :: AbstractLinearOperator=opEye(size(A,1)),
                              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                              itmax :: Int=0, memory :: Int=20, verbose :: Bool=false)

  m, n = size(A)
  m == n || error("System must be square")
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("DIOM: system of size %d\n", n)

  # Initial solution x₀ and residual r₀.
  x = zeros(T, n)
  x_old = copy(x)
  r = copy(b)
  # Compute β.
  rNorm = @knrm2(n, r) # rNorm = ‖r₀‖
  rNorm ≈ 0 && return x, SimpleStats(true, false, [rNorm], [], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  mem = min(memory, itmax) # Memory.
  P = zeros(n, mem) # Directions for x.
  V = zeros(n, mem) # Preconditioned Krylov vectors.
  H = zeros(mem+2)  # Last column of the band hessenberg matrix Hₘ = LₘUₘ.
  # Each column has at most mem + 2 nonzero elements. hᵢ.ₘ is stored as H[m-i+2].
  # m-i+2 represents the indice of the diagonal where hᵢ.ₘ is located.
  # In addition of that, the last column of Uₘ is stored in H.
  L = zeros(mem) # Last mem Pivots of Lₘ.
  p = zeros(Bool, mem) # Last mem permutations.

  # Initial ξ and V.
  ξ = rNorm
  V[:,1] = r / rNorm

  # Stopping criterion.
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index.
    iter = iter + 1

    # Set position in circular stack where iter-th Krylov vector should go.
    pos = mod(iter-1, mem) + 1 # Position corresponding to iter in the circular stack.
    next_pos = mod(iter, mem) + 1 # Position corresponding to iter + 1 in the circular stack.

    # Incomplete Arnoldi procedure.
    z = M * V[:,pos] # Forms pₘ
    w = A * z # Forms vₘ₊₁
    # hₘ₋ₘₑₘ.ₘ = 0
    H[mem+2] = 0
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem) + 1 # Position of vᵢ in the circular stack
      jpos = iter - i + 2
      H[jpos] = @kdot(n, w, V[:,ipos]) # hᵢ.ₘ = < A * vₘ , vᵢ >
      @kaxpy!(n, -H[jpos], V[:,ipos], w) # w ← w - hᵢ.ₘ * vᵢ
    end
    # Compute hₘ₊₁.ₘ and vₘ₊₁.
    H[1] = @knrm2(n, w) # hₘ₊₁.ₘ = ‖vₘ₊₁‖
    if H[1] ≉ 0 # hₘ₊₁.ₘ ≈ 0 ⇒ "lucky breakdown"
      V[:,next_pos] = w / H[1] # vₘ₊₁ = w / hₘ₊₁.ₘ
    end

    # Compute LU factorization with partial pivoting of Hₘ by computing the last column of Uₘ.
    if iter ≥ 2
      for i = max(2,iter-mem+1) : iter
        lpos = mod(i-1, mem) + 1 # Position of lᵢ.ᵢ₋₁ in the circular stack
        jpos = iter - i + 2
        if p[lpos]
          # row i-1 and i are permuted
          H[jpos], H[jpos+1] = H[jpos+1], H[jpos]
        end
        # uᵢ.ₘ ← hᵢ.ₘ - lᵢ.ᵢ₋₁ * uᵢ₋₁.ₘ
        H[jpos] = H[jpos] - L[lpos] * H[jpos+1]
      end
      # Compute ξₘ the last composant of zₘ = β(Lₘ)⁻¹e₁.
      if !p[pos] # p[pos] ⇒ ξₘ = ξₘ₋₁
        # ξₘ = -lₘ.ₘ₋₁ * ξₘ₋₁
        ξ = - L[pos] * ξ
      end
    end

    # Update residual norm estimate.
    # ‖ b - Axₘ ‖ = hₘ₊₁.ₘ * |ξₘ / uₘ.ₘ|
    rNorm = H[1] * abs(ξ / H[2])
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax

    # Compute the direction pₘ, the last column of Pₘ = Vₘ(Uₘ)⁻¹.
    for i = max(1,iter-mem) : iter - 1
      ipos = mod(i-1, mem) + 1
      jpos = iter - i + 2
      # z ← z - uᵢ.ₘ * pᵢ
      @kaxpy!(n, -H[jpos], P[:,ipos], z)
    end

    # Determine if interchange between hₘ₊₁.ₘ and uₘ.ₘ is needed and compute next pivot lₘ₊₁.ₘ.
    if abs(H[2]) < H[1]
      p[next_pos] = true
      # pₘ = z / hₘ₊₁.ₘ
      P[:,pos] = z / H[1]
      # lₘ₊₁.ₘ = uₘ.ₘ / hₘ₊₁.ₘ
      L[next_pos] = H[2] / H[1]
    else
      p[next_pos] = false
      # pₘ = z / uₘ.ₘ
      P[:,pos] = z / H[2]
      # lₘ₊₁.ₘ = hₘ₊₁.ₘ / uₘ.ₘ
      L[next_pos] = H[1] / H[2]
    end

    # Compute solution xₘ.
    if p[pos]
      # xₘ = xₘ₋ₙ + ξₘ₋ₙ * pₘ
      # m-n is the last iteration without permutation at the next step
      x = x_old + ξ * P[:,pos]
    else
      # xₘ = xₘ₋₁ + ξₘ * pₘ
      @kaxpy!(n, ξ, P[:,pos], x)
    end

    # Update x_old.
    if !p[next_pos]
      x_old = copy(x)
    end
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
