# An implementation of CGLS for the solution of the
# over-determined linear least-squares problem
#
#  minimize ‖Ax - b‖
#
# equivalently, of the linear system
#
#  A'Ax = A'b.
#
# This implementation is the standard formulation, as recommended by
# A. Björck, T. Elfving and Z. Strakos, Stability of Conjugate Gradient
# and Lanczos Methods for Linear Least Squares Problems.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

# Todo: allow regularization.

export cgls

type CGLStats <: KrylovStats
  solved :: Bool
  residuals :: Array{Float64,1}
  Aresiduals :: Array{Float64,1}
  status :: UTF8String
end

function show(io :: IO, stats :: CGLStats)
  s  = "\nCGLS stats\n"
  s *= @sprintf("  solved: %s\n", stats.solved)
  s *= @sprintf("  residuals: %7.1e\n", stats.residuals)
  s *= @sprintf("  Aresiduals: %7.1e\n", stats.Aresiduals)
  s *= @sprintf("  status: %s\n", stats.status)
  print(io, s)
end

# Methods for various argument types.
include("cgls_methods.jl")

@doc """
Solve the linear least-squares problem

  minimize ‖b - Ax‖₂

using the Conjugate Gradient (CG) method. This method is equivalent to
applying CG to the normal equations A'Ax = A'b.

CGLS produces monotonic residuals ‖r‖₂ but not optimality residuals ‖A'r‖₂.
It is formally equivalent to LSQR, though can be slightly less accurate,
but simpler to implement.
""" ->
function cgls(A :: LinearOperator, b :: Array{Float64,1};
              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
              verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  x = zeros(n);
  bNorm = BLAS.nrm2(m, b, 1);   # Marginally faster than norm(b);
  bNorm == 0 && return x;
  r = copy(b);
  s = A' * r;
  p = copy(s);
  γ = BLAS.dot(n, s, 1, s, 1);  # Faster than γ = dot(s, s);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm  = bNorm;
  ArNorm = sqrt(γ);
  rNorms = [rNorm];
  ArNorms = [ArNorm];
  ε = atol + rtol * ArNorm;
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 1, ArNorm, rNorm);

  status = "unknown";
  solved = ArNorm <= ε;
  tired = iter >= itmax;

  while ! (solved || tired)
    q = A * p;
    α = γ / BLAS.dot(m, q, 1, q, 1);   # Faster than α = γ / dot(q, q);
    BLAS.axpy!(n,  α, p, 1, x, 1);     # Faster than x = x + α * p;
    BLAS.axpy!(m, -α, q, 1, r, 1);     # Faster than r = r - α * q;
    s = A' * r;
    γ_next = BLAS.dot(n, s, 1, s, 1);  # Faster than γ_next = dot(s, s);
    β = γ_next / γ;
    BLAS.scal!(n, β, p, 1);
    BLAS.axpy!(n, 1.0, s, 1, p, 1);    # Faster than p = s + β * p;
    # The combined BLAS calls tend to trigger some gc.
    #  BLAS.axpy!(n, 1.0, s, 1, BLAS.scal!(n, β, p, 1), 1);
    γ = γ_next;
    rNorm = BLAS.nrm2(m, r, 1);  # Marginally faster than norm(r);
    ArNorm = sqrt(γ);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 1 + 2 * iter, ArNorm, rNorm);
    solved = ArNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = CGLStats(solved, rNorms, ArNorms, status);
  return (x, stats);
end
