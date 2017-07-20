# A truncated version of Stiefel’s Conjugate Residual method
# cr(A, b, atol, rtol, itmax, radius, verbose) solves the linear system 'A * x = b' or the least-squares problem :
# 'min ‖b - A * x‖²' within a region of fixed radius.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

export cr

"""A truncated version of Stiefel’s Conjugate Residual method to solve the symmetric linear system Ax=b.
The matrix A must be positive semi-definite
"""
function cr{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T}, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, radius :: Float64=0., verbose :: Bool=true)

  n = size(b, 1) # size of the problem
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CR: system of %d equations in %d variables\n", n, n)

  # Initial state.
  x = zeros(T, n) # initial estimation x = 0
  r = copy(b) # initial residual r = b - Ax = b
  Ar = A * r
  ρ = @kdot(n, r, Ar)
  ρ == 0.0 && return (x, Krylov.SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution"))
  p = copy(r)
  q = copy(Ar)

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  rNorm = @knrm2(n, r) # ‖r‖
  rNorms = [rNorm] # Values of ‖r‖
  ArNorm = @knrm2(n, Ar) # ‖Ar‖
  ArNorms = [ArNorm]
  ε = atol + rtol * rNorm
  verbose && @printf("%5s %8s %5s %8s\n", "Iter", "‖r‖", "α", "σ")
  verbose && @printf("    %d  %8.1e", iter, rNorm)

  solved = rNorm <= ε
  tired = iter >= itmax
  on_boundary = false
  status = "unknown"

  while ! (solved || tired)
    α = ρ / @kdot(n, q, q) # step

    # Compute step size to boundary if applicable.
    σ = radius > 0.0 ? to_boundary(x, p, radius) : α

    verbose && @printf("  %7.1e   %7.1e\n", α, σ);

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0.0) & (α > σ)
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α,  p, x)
    @kaxpy!(n, -α, q, r) # residual
    rNorm = @knrm2(n, r)
    push!(rNorms, rNorm)
    Ar = A * r
    ArNorm = @knrm2(n, Ar)
    push!(ArNorms, ArNorm)
    
    iter = iter + 1
    verbose && @printf("    %d  %8.1e", iter, rNorm)

    solved = (rNorm <= ε) | on_boundary
    tired = iter >= itmax
    
    (solved || tired) && continue
    ρbar = ρ
    ρ = @kdot(n, r, Ar)
    β = ρ / ρbar # step for the direction computation
    @kaxpy!(n, 1.0, r, β, p)
    @kaxpy!(n, 1.0, Ar, β, q)

  end
  verbose && @printf("\n")

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = Krylov.SimpleStats(solved, false, rNorms, ArNorms, status)
  return (x, stats)
end
