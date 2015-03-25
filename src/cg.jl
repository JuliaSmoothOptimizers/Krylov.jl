export cg

# Methods for various argument types.
include("cg_methods.jl")


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
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e\n", iter, rNorm);

  while (rNorm > ε) & (iter < itmax)
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
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e\n", iter, rNorm);
  end
  return x;
end
