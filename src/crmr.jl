# An implementation of CRMR for the solution of the
# (under/over-determined or square) linear system
#
#  Ax = b.
#
# The method seeks to solve the minimum-norm problem
#
#  min ‖x‖²  s.t. Ax = b,
#
# and is equivalent to applying the conjugate residual method
# to the linear system
#
#  AA'y = b.
#
# This method is equivalent to Craig-MR, described in
#
# M. Arioli and D. Orban, Iterative Methods for Symmetric
# Quasi-Definite Linear Systems, Part I: Theory.
# Cahier du GERAD G-2013-32, GERAD, Montreal QC, Canada, 2013.
#
# D. Orban, The Projected Golub-Kahan Process for Constrained
# Linear Least-Squares Problems. Cahier du GERAD G-2014-15,
# GERAD, Montreal QC, Canada, 2014.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montreal, QC, April 2015.

export crmr


"""Solve the consistent linear system

  Ax + √λs = b

using the Conjugate Residual (CR) method, where λ ≥ 0 is a regularization
parameter. This method is equivalent to applying CR to the normal equations
of the second kind

  (AA' + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

  min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

  min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRAIG-MR, though can be slightly less accurate,
but simpler to implement. Only the x-part of the solution is returned.
"""
function crmr{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T};
                           λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                           itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CRMR: system of %d equations in %d variables\n", m, n);

  x = zeros(T, n)
  r  = copy(b)
  bNorm = @knrm2(m, r)  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x, SimpleStats(true, false, [0.0], [0.0], "x = 0 is a zero-residual solution");
  rNorm = bNorm;  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  λ > 0 && (s = copy(r));
  Ar = A' * r;  # - λ * x0 if x0 ≠ 0.
  p  = copy(Ar);
  γ  = @kdot(n, Ar, Ar)  # Faster than γ = dot(Ar, Ar);
  λ > 0 && (γ += λ * rNorm * rNorm);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  ArNorm = sqrt(γ);
  rNorms = [rNorm;];
  ArNorms = [ArNorm;];
  ɛ_c = atol + rtol * rNorm;   # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm;  # Stopping tolerance for inconsistent systems.
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm);

  status = "unknown";
  solved = rNorm <= ɛ_c;
  inconsistent = (rNorm > 1.0e+2 * ɛ_c) && (ArNorm <= ɛ_i);
  tired = iter >= itmax;

  while ! (solved || inconsistent || tired)
    q = A * p;
    λ > 0 && @kaxpy!(m, λ, s, q)  # q = q + λ * s;
    α = γ / @kdot(m, q, q)     # dot(q, q);
    @kaxpy!(n,  α, p, x)       # Faster than  x =  x + α *  p;
    @kaxpy!(m, -α, q, r)       # Faster than  r =  r - α * Ap;
    rNorm = @knrm2(m, r)       # norm(r);
    Ar = A' * r;
    γ_next = @kdot(n, Ar, Ar)  # Faster than γ_next = dot(Ar, Ar);
    λ > 0 && (γ_next += λ * rNorm * rNorm);
    β = γ_next / γ;

    @kscal!(n, β, p)
    @kaxpy!(n, 1.0, Ar, p)    # Faster than  p = Ar + β *  p;
    if λ > 0
      @kscal!(m, β, s)
      @kaxpy!(m, 1.0, r, s)   # s = r + β * s;
    end

    γ = γ_next;
    ArNorm = sqrt(γ);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm);
    solved = rNorm <= ɛ_c;
    inconsistent = (rNorm > 1.0e+2 * ɛ_c) && (ArNorm <= ɛ_i);
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent but least squares/norm solution found" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status);
  return (x, stats);
end
