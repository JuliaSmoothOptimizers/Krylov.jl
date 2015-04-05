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

export crls

# Methods for various argument types.
include("crls_methods.jl")

@doc """
Solve the linear least-squares problem

  minimize ‖b - Ax‖₂² + λ ‖x‖₂²

using the Conjugate Residuals (CR) method. This method is equivalent to
applying MINRES to the normal equations

  (A'A + λI) x = A'b.

This implementation recurs the residual r := b - Ax.

CRLS produces monotonic residuals ‖r‖₂ and optimality residuals ‖A'r‖₂.
It is formally equivalent to LSMR, though can be slightly less accurate,
but simpler to implement.
""" ->
function crls(A :: LinearOperator, b :: Array{Float64,1};
              λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
              itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  x = zeros(n);
  bNorm = BLAS.nrm2(m, b, 1);  # norm(b - A * x0) if x0 ≠ 0.
  bNorm == 0 && return x;
  r  = copy(b);

  # The following vector copy takes care of the case where A is a LinearOperator
  # with preallocation, so as to avoid overwriting vectors used later. In other
  # case, this should only add minimum overhead.
  Ar = copy(A' * b);  # - λ * x0 if x0 ≠ 0.

  s  = A * Ar;
  p  = copy(Ar);
  Ap = copy(s);
  q  = A' * Ap;
  λ > 0 && BLAS.axpy!(n, λ, p, 1, q, 1);  # q = q + λ * p;
  γ  = BLAS.dot(m, s, 1, s, 1);  # Faster than γ = dot(s, s);
  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm = bNorm;  # + λ * ‖x0‖ if x0 ≠ 0 and λ > 0.
  ArNorm = BLAS.nrm2(n, Ar, 1);  # Marginally faster than norm(Ar);
  λ > 0 && (γ += λ * ArNorm * ArNorm);
  rNorms = [rNorm];
  ArNorms = [ArNorm];
  ε = atol + rtol * ArNorm;
  verbose && @printf("%5s  %8s  %8s\n", "Aprod", "‖A'r‖", "‖r‖")
  verbose && @printf("%5d  %8.2e  %8.2e\n", 3, ArNorm, rNorm);

  status = "unknown";
  solved = ArNorm <= ε;
  tired = iter >= itmax;

  while ! (solved || tired)
    α = γ / BLAS.dot(n, q, 1, q, 1);     # dot(q, q);
    BLAS.axpy!(n,  α, p,  1,  x, 1);     # Faster than  x =  x + α *  p;
    BLAS.axpy!(m, -α, Ap, 1,  r, 1);     # Faster than  r =  r - α * Ap;
    BLAS.axpy!(n, -α, q,  1, Ar, 1);     # Faster than Ar = Ar - α *  q;
    s = A * Ar;
    γ_next = BLAS.dot(m, s, 1, s, 1);   # Faster than γ_next = dot(s, s);
    ArNorm = BLAS.nrm2(n, Ar, 1);
    λ > 0 && (γ_next += λ * ArNorm * ArNorm);
    β = γ_next / γ;

    BLAS.scal!(n, β, p, 1);
    BLAS.axpy!(n, 1.0, Ar, 1, p, 1);    # Faster than  p = Ar + β *  p;
    # The combined call uses less memory but tends to trigger more gc.
    #     BLAS.axpy!(n, 1.0, Ar, 1, BLAS.scal!(n, β, p, 1), 1);

    BLAS.scal!(m, β, Ap, 1);
    BLAS.axpy!(m, 1.0, s, 1, Ap, 1);    # Faster than Ap =  s + β * Ap;
    q = A' * Ap;
    λ > 0 && BLAS.axpy!(n, λ, p, 1, q, 1);  # q = q + λ * p;

    γ = γ_next;
    if λ > 0
      rNorm = sqrt(BLAS.dot(m, r, 1, r, 1) + λ * BLAS.dot(n, x, 1, x, 1));
    else
      rNorm = BLAS.nrm2(m, r, 1);  # norm(r);
    end
    #     ArNorm = norm(Ar);
    push!(rNorms, rNorm);
    push!(ArNorms, ArNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.2e  %8.2e\n", 3 + 2 * iter, ArNorm, rNorm);
    solved = ArNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = SimpleStats(solved, false, rNorms, ArNorms, status);
  return (x, stats);
end
