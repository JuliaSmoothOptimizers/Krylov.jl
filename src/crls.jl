# An implementation of CRLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the linear system
#
#  A'Ax = A'b.
#
# This implementation follows the formulation given in
# D. C.-L. Fong, Minimum-Residual Methods for Sparse
# Least-Squares using Golubg-Kahan Bidiagonalization,
# Ph.D. Thesis, Stanford University, 2011.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

# Todo: allow regularization.

export crls

# Methods for various argument types.
include("crls_methods.jl")


function crls(A :: LinearOperator, b :: Array{Float64,1};
              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
              verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  x = zeros(n);
  norm(b) == 0 && return x;
  r = A' * b;
  s = A * r;
  p = copy(r);
  q = A' * s;
  γ = BLAS.dot(m, s, 1, s, 1);  # Faster than γ = dot(s, s);
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  rNorm = sqrt(γ);
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e\n", iter, rNorm);

  while (rNorm > ε) & (iter < itmax)
    α = γ / dot(q, q);
    BLAS.axpy!(n,  α, p, 1, x, 1);     # Faster than x = x + α * p;
    BLAS.axpy!(n, -α, q, 1, r, 1);     # Faster than r = r - α * q;
    s = A * r;
    γ_next = BLAS.dot(m, s, 1, s, 1);  # Faster than γ_next = dot(s, s);
    β = γ_next / γ;
    BLAS.scal!(n, β, p, 1);
    BLAS.axpy!(n, 1.0, r, 1, p, 1);    # Faster than p = r + β * p;
    # The combined call uses less memory but tends to trigger more gc.
    #     BLAS.axpy!(n, 1.0, r, 1, BLAS.scal!(n, β, p, 1), 1);
    q = A' * s + β * q;
    # The BLAS calls are not faster here and trigger lots of gc.
    #     BLAS.scal!(n, β, q, 1);
    #     BLAS.axpy!(n, 1.0, A' * s, 1, q, 1);
    #     BLAS.axpy!(n, 1.0, A' * s, 1, BLAS.scal!(n, β, q, 1), 1);
    γ = γ_next;
    rNorm = sqrt(γ);
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e\n", iter, rNorm);
  end
  return x;
end
