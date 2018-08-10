# An implementation of DQGMRES for the solution of the square linear system Ax = b.
#
# This method is described in
#
# Y. Saad, Iterative methods for sparse linear systems.
# PWS Publishing Company, Boston, USA, 1996.
#
# Y. Saad and K. Wu, DQGMRES: a quasi minimal residual algorithm based on incomplete orthogonalization.
# Numerical Linear Algebra with Applications, Vol. 3(4), 329-343, 1996.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, August 2018.

export dqgmres

"""Solve the consistent linear system Ax = b using DQGMRES method.

DQGMRES algorithm is based on the incomplete Arnoldi orthogonalization process
and computes a sequence of approximate solutions with the quasi-minimal residual property.

This implementation allows a right preconditioning with M.
"""
function dqgmres{T <: Number}(A :: AbstractLinearOperator, b :: AbstractVector{T};
                              M :: AbstractLinearOperator=opEye(size(A,1)),
                              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                              itmax :: Int=0, memory :: Int=20, verbose :: Bool=false)

  m, n = size(A)
  m == n || error("System must be square")
  size(b, 1) == m || error("Inconsistent problem size")
  verbose && @printf("DQGMRES: system of size %d\n", n)

  # Initial solution and residual.
  x = zeros(T, n)
  r = copy(b)
  rNorm = @knrm2(n, r)
  rNorm ≈ 0 && return x, SimpleStats(true, false, [rNorm], [], "x = 0 is a zero-residual solution")

  iter = 0
  itmax == 0 && (itmax = 2*n)

  rNorms = [rNorm;]
  ε = atol + rtol * rNorm
  verbose && @printf("%5d  %7.1e\n", iter, rNorm)

  # Set up workspace.
  mem = min(memory, itmax) # Memory.
  g = zeros(mem+1) # Right-hand of the least squares problem min ‖ Hy - g ‖.
  V = zeros(n, mem+1) # Preconditioned Krylov vectors.
  P = zeros(n,mem+1) # Directions for x: P = V * inv(R).
  s = zeros(mem) # Givens sines.
  c = zeros(mem) # Givens cosines.
  H = spzeros(itmax, mem+2) # Upper Hessenberg matrix. Its mem+2 diagonals are stored
                            # as column vectors, according to the tranformation
                            # (j,k) --> (j,2+k-j).
  # Initial g and V.
  g[1] = rNorm
  V[:,1] = r / rNorm

  # The following stopping criterion compensates for the lag in the
  # residual, but usually increases the number of iterations.
  # solved = sqrt(max(1, iter-mem+1)) <= ε
  solved = rNorm <= ε # less accurate, but acceptable.
  tired = iter >= itmax
  status = "unknown"

  while !(solved || tired)

    # Update iteration index.
    iter = iter + 1

    # Set position in circular stack where iter-th Krylov vector should go.
    pos = mod(iter-1, mem+1) + 1 # Position corresponding to iter in the circular stack.
    next_pos = mod(iter, mem+1) + 1 # Position corresponding to iter+1 in the circular stack.
    rot_pos = mod(iter-1, mem) + 1 # Position of the rotation Ω generate at current iteration.

    # Incomplete Arnoldi procedure.
    z = M * V[:,pos] # P[:,pos]
    w = A * z # V[:,next_pos]
    for i = max(1, iter-mem+1) : iter
      ipos = mod(i-1, mem+1) + 1
      jpos = iter-i+2 # Indice of the diagonal
      H[i,jpos] = @kdot(n, w, V[:,ipos]) # H[i.jpos] = < w , V[:,ipos] >
      @kaxpy!(n, -H[i,jpos], V[:,ipos], w) # w = w - H[i,jpos] * V[:,ipos]
    end
    # jpos = iter-(iter+1)+2 = 1
    H[iter,1] = @knrm2(n, w)
    if H[iter,1] ≉ 0 # if H[iter,1] ≈ 0 => "lucky breakdown"
      V[:,next_pos] = w / H[iter,1]
    end

    # Update the QR factorization of H.
    # Apply mem previous (symmetric) Givens rotations.
    for i = max(1,iter-mem) : iter-1
      ipos = mod(i-1, mem+1) + 1
      ip1pos = mod(i, mem+1) + 1
      irot_pos = mod(i-1, mem) + 1
      jpos = iter - i + 1 # jpos = 2+iter-(i+1)
      jp1pos = jpos + 1   # jp1pos = 2+iter-i
      H_aux       = c[irot_pos] * H[i,jp1pos] + s[irot_pos] * H[i+1,jpos]
      H[i+1,jpos] = s[irot_pos] * H[i,jp1pos] - c[irot_pos] * H[i+1,jpos]
      H[i,jp1pos] = H_aux
    end

    # Compute and apply current (symmetric) Givens rotation
    # [ck  sk] [ H[iter,iter]   ] = [ρ]
    # [sk -ck] [ H[iter+1,iter] ]   [0].
    (c[rot_pos], s[rot_pos], H[iter,2]) = sym_givens(H[iter,2], H[iter,1])
    H[iter,1]   = 0
    g[next_pos] = s[rot_pos] * g[pos]
    g[pos]      = c[rot_pos] * g[pos]

    # Update directions P and solution x
    #P[:,pos] = z
    for i = max(1,iter-mem) : iter-1
      ipos = mod(i-1, mem+1) + 1
      jpos = iter - i + 2
      @kaxpy!(n, -H[i,jpos], P[:,ipos], z) # z = z - H[i,jpos] * P[:,ipos]
    end
    P[:,pos] = z / H[iter,2]
    @kaxpy!(n, g[pos], P[:,pos], x) # x = x + g[pos] * P[:,pos]

    # Update residual norm estimate.
    rNorm = abs(g[next_pos])
    push!(rNorms, rNorm)

    # Update stopping criterion.
    solved = rNorm <= ε
    tired = iter >= itmax
    verbose && @printf("%5d  %7.1e\n", iter, rNorm)
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, T[], status)
  return (x, stats)
end
