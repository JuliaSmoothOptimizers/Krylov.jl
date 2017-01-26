# A standard implementation of the Conjugate Gradient method.
# The only non-standard point about it is that it does not check
# that the operator is definite.
# It is possible to check that the system is inconsistent by
# monitoring ‖p‖, which would cost an extra norm computation per
# iteration.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Salt Lake City, UT, March 2015.

export cg


"""The conjugate gradient method to solve the symmetric linear system Ax=b.

The method does _not_ abort if A is not definite.
"""
function cg{T <: Number}(A :: AbstractLinearOperator, b :: Vector{T};
                         atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                         radius :: Float64=0.0, verbose :: Bool=false)

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");
  verbose && @printf("CG: system of %d equations in %d variables\n", n, n);

  # Initial state.
  x = zeros(T, n);
  γ = @kdot(n, b, b)
  γ == 0 && return x, SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution")
  r = copy(b)
  p = copy(r);

  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  rNorm = sqrt(γ);
  rNorms = [rNorm;];
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e  ", iter, rNorm);

  solved = rNorm <= ε;
  tired = iter >= itmax;
  on_boundary = false;
  status = "unknown";

  while ! (solved || tired)
    Ap = A * p;
    pAp = @kdot(n, p, Ap)

    α = γ / pAp;

    # Compute step size to boundary if applicable.
    σ = radius > 0.0 ? to_boundary(x, p, radius) : α

    verbose && @printf("%8.1e  %7.1e  %7.1e\n", pAp, α, σ);

    # Move along p from x to the boundary if either
    # the next step leads outside the trust region or
    # we have nonpositive curvature.
    if (radius > 0.0) & ((pAp <= 0.0) | (α > σ))
      α = σ
      on_boundary = true
    end

    @kaxpy!(n,  α,  p, x)
    @kaxpy!(n, -α, Ap, r)
    γ_next = @kdot(n, r, r)
    rNorm = sqrt(γ_next);
    push!(rNorms, rNorm);

    solved = (rNorm <= ε) | on_boundary;
    tired = iter >= itmax;

    if !solved
      β = γ_next / γ;
      γ = γ_next;

      @kscal!(n, β, p)
      @kaxpy!(n, 1.0, r, p)
    end
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e  ", iter, rNorm);
  end
  verbose && @printf("\n");

  status = on_boundary ? "on trust-region boundary" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, T[], status);
  return (x, stats);
end
