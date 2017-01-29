# An implementation of CGLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the normal equations
#
#  A'Ax = A'b.
#
# CGLS is formally equivalent to applying the conjugate gradient method
# to the normal equations but should be more stable. It is also formally
# equivalent to LSQR though LSQR should be expected to be more stable on
# ill-conditioned or poorly scaled problems.
#
# This implementation is the standard formulation, as recommended by
# A. Björck, T. Elfving and Z. Strakos, Stability of Conjugate Gradient
# and Lanczos Methods for Linear Least Squares Problems.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cgls


"""Solve the regularized linear least-squares problem

  minimize ‖b - Ax‖₂² + λ ‖x‖₂²

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations

  (A'A + λI) x = A'b

but is more stable.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖A'r‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.
"""
function cgls{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T};
                           M :: AbstractLinearOperator=opEye(size(b,1)),
                           λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                           itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CGLS: system of %d equations in %d variables\n", m, n);

  x = zeros(T, n);
  r = copy(b)
  bNorm = @knrm2(m, r)   # Marginally faster than norm(b);
  bNorm == 0 && return x, SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution");
  s = A' * M * r;
  p = copy(s);
  γ = @kdot(n, s, s)  # Faster than γ = dot(s, s);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm  = bNorm;
  ArNorm = sqrt(γ);
  rNorms = [rNorm;];
  ArNorms = [ArNorm;];
  ε = atol + rtol * ArNorm;
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm);

  status = "unknown";
  solved = ArNorm <= ε;
  tired = iter >= itmax;

  while ! (solved || tired)
    q = A * p;
    δ = @kdot(m, q, M * q)   # Faster than α = γ / dot(q, q);
    λ > 0 && (δ += λ * @kdot(n, p, p))
    α = γ / δ;
    @kaxpy!(n,  α, p, x)     # Faster than x = x + α * p;
    @kaxpy!(m, -α, q, r)     # Faster than r = r - α * q;
    s = A' * M * r;
    λ > 0 && @kaxpy!(n, -λ, x, s)   # s = A' * r - λ * x;
    γ_next = @kdot(n, s, s)  # Faster than γ_next = dot(s, s);
    β = γ_next / γ;
    @kscal!(n, β, p)
    @kaxpy!(n, 1.0, s, p)    # Faster than p = s + β * p;
    # The combined BLAS calls tend to trigger some gc.
    #  BLAS.axpy!(n, 1.0, s, 1, BLAS.scal!(n, β, p, 1), 1);
    γ = γ_next;
    rNorm = @knrm2(m, r)  # Marginally faster than norm(r);
    ArNorm = sqrt(γ);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm);
    solved = ArNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, ArNorms, status);
  return (x, stats);
end
