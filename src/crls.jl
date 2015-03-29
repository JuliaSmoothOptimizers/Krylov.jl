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
#
# D. C.-L. Fong, Minimum-Residual Methods for Sparse
# Least-Squares using Golubg-Kahan Bidiagonalization,
# Ph.D. Thesis, Stanford University, 2011
#
# with the difference that it also recurs r = b - Ax.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

# Todo: allow regularization.

export crls

type CRLStats <: KrylovStats
  solved :: Bool
  residuals :: Array{Float64,1}
  Aresiduals :: Array{Float64,1}
  status :: UTF8String
end

# Methods for various argument types.
include("crls_methods.jl")


function crls(A :: LinearOperator, b :: Array{Float64,1};
              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
              verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  x = zeros(n);
  bNorm = norm(b);
  bNorm == 0 && return x;
  r  = copy(b);
  Ar = A' * b;
  s  = A * Ar;
  p  = copy(Ar);
  Ap = copy(s);
  q  = A' * Ap;
  γ  = BLAS.dot(m, s, 1, s, 1);  # Faster than γ = dot(s, s);
  iter = 0;
  itmax == 0 && (itmax = min(m, n, 20));

  rNorm = bNorm;
  ArNorm = norm(Ar);
  rNorms = [rNorm];
  ArNorms = [ArNorm];
  ε = atol + rtol * ArNorm;
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 3, ArNorm, rNorm);

  status = "unknown";
  solved = ArNorm <= ε;
  tired = iter >= itmax;

  while ! (solved || tired)
    α = γ / dot(q, q);
    BLAS.axpy!(n,  α, p,  1,  x, 1);     # Faster than  x =  x + α *  p;
    BLAS.axpy!(m, -α, Ap, 1,  r, 1);     # Faster than  r =  r - α * Ap;
    BLAS.axpy!(n, -α, q,  1, Ar, 1);     # Faster than Ar = Ar - α *  q;
    s = A * Ar;
    γ_next = BLAS.dot(m, s, 1, s, 1);   # Faster than γ_next = dot(s, s);
    β = γ_next / γ;

    BLAS.scal!(n, β, p, 1);
    BLAS.axpy!(n, 1.0, Ar, 1, p, 1);    # Faster than  p = Ar + β *  p;
    # The combined call uses less memory but tends to trigger more gc.
    #     BLAS.axpy!(n, 1.0, Ar, 1, BLAS.scal!(n, β, p, 1), 1);

    BLAS.scal!(m, β, Ap, 1);
    BLAS.axpy!(m, 1.0, s, 1, Ap, 1);    # Faster than Ap =  s + β * Ap;
    q = A' * Ap;

    γ = γ_next;
    rNorm = norm(r);
    ArNorm = norm(Ar);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 3 + 2 * iter, ArNorm, rNorm);
    solved = ArNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = CRLStats(solved, rNorms, ArNorms, status);
  return (x, stats);
end
