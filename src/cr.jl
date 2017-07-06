# A truncated version of Stiefel’s Conjugate Residual method
# cr(A, b, Δ, rtol, itmax, verbose) solves the linear system 'A * x = b' or the least-squares problem :
# 'min ‖b - A * x‖²' within a region of fixed radius Δ.
#
# Marie-Ange Dahito, <marie-ange.dahito@polymtl.ca>
# Montreal, QC, June 2017

# export cr

"""A truncated version of Stiefel’s Conjugate Residual method to solve the symmetric linear system Ax=b.
The matrix A must be semi positive definite
"""
function cr{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T}, Δ :: Float64=10., rtol :: Float64=1.0e-6, itmax :: Int=0, verbose :: Bool=true)

  n = size(b, 1) # size of the problem
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")
  verbose && @printf("CR: system of %d equations in %d variables\n", n, n)

  x = zeros(T, n) # initial estimation x = 0
  r = copy(b) # initial residual r = b - Ax = b
  r == 0.0 && return (x, SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution"))
  Ar = A * r
  ρ = @kdot(n, r, Ar)
  p = copy(r)
  q = copy(Ar)
  Δ² = Δ^2

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  mvalues = [0.0] # values of the quadratic model
  xNorms = [0.0] # Values of ‖x‖
  rNorm = @knrm2(n, r) # ‖r‖
  rNorms = [rNorm] # Values of ‖r‖
  ArNorm = @knrm2(n, Ar) # ‖Ar‖
  ArNorms = [ArNorm]
  verbose && @printf("%5s %5s %10s %10s\n", "Iter", "‖x‖", "‖r‖", "q")

  on_boundary = false
  status = "unknown"
  solved = (rNorm <= rtol)
  tired = iter >= itmax

  while ! (solved || tired)
    iter = iter + 1

    # solve ‖x+t1*p‖² - Δ² = 0 with t1 >= 0
    c = @knrm2(n, p)^2
    a = @kdot(n, x, p)
    f = @knrm2(n, x)^2 - Δ²
    t = sqrt(a^2 - c * f)

    # Compute t1 with reduced numerical errors
    if a < 0.0
      t1 = (-a + t) / c
    else
      t1 = f / (-a - t)
    end

    α = ρ / @knrm2(n, q)^2 # step

    # if x is out of the trust region, p is followed until the edge of the trust region
		if α >= t1
			@kaxpy!(n, t1, p, x)
      xNorm = @knrm2(n, x)
      xNorms = push!(xNorms, xNorm)
      m = @kdot(n, -b, x) + 1/2 * @kdot(n, x, A * x)
      mvalues = push!(mvalues, m)
      rNorm = @knrm2(n, A * x - b)
      rNorms = push!(rNorms, rNorm)
      ArNorm = @knrm2(n, Ar)
      ArNorms = push!(ArNorms, ArNorm)
      on_boundary = true
      solved = (rNorm <= rtol) | on_boundary
      status = "on_boundary"
      stats = SimpleStats(solved, false, rNorms, ArNorms, status)
      verbose && @printf("%d    %8.1e    %8.1e    %8.1e\n", iter, xNorm, rNorm, m)
			return (x, stats)
		end

    @kaxpy!(n, α, p, x) # new estimation
    xNorm = @knrm2(n, x)
    xNorms = push!(xNorms, xNorm)
    m = @kdot(n, -b, x) + 1/2 * @kdot(n, x, A * x)
    mvalues = push!(mvalues, m)
    @kaxpy!(n, -α, q, r) # residual
    Ar = A * r
    ρbar = ρ
    ρ = @kdot(n, r, Ar)
    β = ρ / ρbar # step for the direction calculus
    p = r + β * p # descent direction
    q = Ar + β * q
    rNorm = @knrm2(n, r) # ‖r‖
    rNorms = push!(rNorms, rNorm)
    ArNorm = @knrm2(n, Ar)
    ArNorms = push!(ArNorms, ArNorm)
    solved = rNorm <= rtol
    tired = iter >= itmax

    verbose && @printf("%d    %8.1e    %8.1e    %8.1e\n", iter, xNorm, rNorm, m)

  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given rtol"
  stats = SimpleStats(solved, false, rNorms, ArNorms, status)

  return (x, stats)
end
