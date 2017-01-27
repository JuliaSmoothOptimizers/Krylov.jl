# An implementation of CGNE for the solution of the consistent
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate gradient method
# to the linear system
#
#  AA'y = b.
#
# This method is also known as Craig's method, CGME, and other
# names, and is described in
#
# J. E. Craig. The N-step iteration procedures.
# Journal of Mathematics and Physics, 34(1):64--73, 1955.
#
# which is based on Craig's thesis from MIT:
#
# J. E. Craig. Iterations Procedures for Simultaneous Equations.
# Ph.D. Thesis, Department of Electrical Engineering, MIT, 1954.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.

export cgne


"""Solve the consistent linear system

  Ax + √λs = b

using the Conjugate Gradient (CG) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CG to the normal equations
of the second kind

  (AA' + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

  min ‖x‖₂  s.t. Ax = b.

When λ > 0, it solves the problem

  min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGNE produces monotonic errors ‖x-x*‖₂ but not residuals ‖r‖₂.
It is formally equivalent to CRAIG, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.
"""
function cgne{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T};
                           λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                           itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CGNE: system of %d equations in %d variables\n", m, n);

  x = zeros(T, n);
  r = copy(b)
  bNorm = @knrm2(m, r)   # Marginally faster than norm(b);
  bNorm == 0 && return x, SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution");
  λ > 0 && (s = copy(r));

  # The following vector copy takes care of the case where A is a LinearOperator
  # with preallocation, so as to avoid overwriting vectors used later. In other
  # case, this should only add minimum overhead.
  p = copy(A' * r);

  # Use ‖p‖ to detect inconsistent system.
  # An inconsistent system will necessarily have AA' singular.
  # Because CGNE is equivalent to CG applied to AA'y = b, there will be a
  # conjugate direction u such that u'AA'u = 0, i.e., A'u = 0. In this
  # implementation, p is a substitute for A'u.
  pNorm = @knrm2(n, p)

  γ = @kdot(m, r, r)  # Faster than γ = dot(r, r);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm  = bNorm;
  rNorms = [rNorm;];
  ɛ_c = atol + rtol * rNorm;  # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * pNorm;  # Stopping tolerance for inconsistent systems.
  verbose && @printf("%5s  %8s\n", "Aprod", "‖r‖")
  verbose && @printf("%5d  %8.2e\n", 1, rNorm);

  status = "unknown";
  solved = rNorm <= ɛ_c;
  inconsistent = (rNorm > 1.0e+2 * ɛ_c) && (pNorm <= ɛ_i);
  tired = iter >= itmax;

  while ! (solved || inconsistent || tired)
    q = A * p;
    λ > 0 && @kaxpy!(m, λ, s, q)
    δ = @kdot(n, p, p)   # Faster than dot(p, p);
    λ > 0 && (δ += λ * @kdot(m, s, s))
    α = γ / δ;
    @kaxpy!(n,  α, p, x)     # Faster than x = x + α * p;
    @kaxpy!(m, -α, q, r)     # Faster than r = r - α * q;
    γ_next = @kdot(m, r, r)  # Faster than γ_next = dot(r, r);
    β = γ_next / γ;
    @kscal!(n, β, p)
    @kaxpy!(n, 1.0, A' * r, p)   # Faster than p = A' * r + β * p;
    pNorm = @knrm2(n, p)
    if λ > 0
      @kscal!(m, β, s)
      @kaxpy!(m, 1.0, r, s)   # s = r + β * s;
    end
    γ = γ_next;
    rNorm = sqrt(γ_next);
    push!(rNorms, rNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e\n", 1 + 2 * iter, rNorm);
    solved = rNorm <= ɛ_c;
    inconsistent = (rNorm > 1.0e+2 * ɛ_c) && (pNorm <= ɛ_i);
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, [], status);
  return (x, stats);
end
