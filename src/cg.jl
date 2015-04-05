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

# Methods for various argument types.
include("cg_methods.jl")

@doc """
The conjugate gradient method to solve the symmetric linear system Ax=b.

The method does _not_ abort if A is not definite.
""" ->
function cg{T <: Real}(A :: LinearOperator, b :: Array{T,1};
                       atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                       verbose :: Bool=false)

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");

  # Initial state.
  x = zeros(n);
  γ = dot(b, b);
  γ == 0 && return x;
  r = copy(b);
  p = copy(r);

  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  rNorm = sqrt(γ);
  rNorms = [rNorm];
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e\n", iter, rNorm);

  solved = rNorm <= ε;
  tired = iter >= itmax;
  status = "unknown";

  while ! (solved || tired)
    Ap = A * p;
    pAp = BLAS.dot(n, p, 1, Ap, 1);
    α = γ / pAp;
    BLAS.axpy!(n,  α,  p, 1, x, 1);  # Faster than x = x + α * p;
    BLAS.axpy!(n, -α, Ap, 1, r, 1);  # Faster than r = r - α * Ap;
    γ_next = BLAS.dot(n, r, 1, r, 1);
    β = γ_next / γ;
    BLAS.scal!(n, β, p, 1)
    BLAS.axpy!(n, 1.0, r, 1, p, 1);  # Faster than p = r + β * p;
    γ = γ_next;
    rNorm = sqrt(γ);
    push!(rNorms, rNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e\n", iter, rNorm);
    solved = rNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, [], status);
  return (x, stats);
end
