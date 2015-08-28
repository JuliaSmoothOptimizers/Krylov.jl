# An implementation of the Golub-Kahan version of Craig's method
# for the solution of the consistent (under/over-determined or square)
# linear system
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
# This method, sometimes known under the name CRAIG, is the
# Golub-Kahan implementation of CGNE, and is described in
#
# C. C. Paige and M. A. Saunders, LSQR: An Algorithm for Sparse
# Linear Equations and Sparse Least Squares, ACM Transactions on
# Mathematical Software, Vol 8, No. 1, pp. 43-71, 1982.
#
# and
#
# M. A. Saunders, Solutions of Sparse Rectangular Systems Using
# LSQR and CRAIG, BIT, No. 35, pp. 588-604, 1995.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Montréal, QC, April 2015.
#
# This implementation is strongly inspired from Mike Saunders's.

export craig


# Methods for various argument types.
include("craig_methods.jl")

"""Find the least-norm solution of the consistent linear system

  Ax + √λs = b

using the Golub-Kahan implementation of Craig's method, where λ ≥ 0 is a
regularization parameter. This method is equivalent to CGNE but is more
stable.

For a system in the form Ax = b, Craig's method is equivalent to applying
CG to AA'y = b and recovering x = A'y. Note that y are the Lagrange
multipliers of the least-norm problem

  minimize ‖x‖  subject to Ax = b.

In this implementation, both the x and y-parts of the solution are returned.
"""
function craig(A :: LinearOperator, b :: Array{Float64,1};
               λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               conlim :: Float64=1.0e+8, itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CRAIG: system of %d equations in %d variables\n", m, n);

  x = zeros(n);
  β₁ = BLAS.nrm2(m, b, 1);   # Marginally faster than norm(b);
  β₁ == 0 && return x;
  β = β₁;
  θ = β₁;   # θ will differ from β when there is regularization (λ > 0).
  ξ = -1;   # Most recent component of x in Range(V).
  δ = λ;
  ρ_prev = 1;

  # β₁ u₁ = b.
  u = copy(b);
  BLAS.scal!(m, 1.0/β₁, u, 1);

  v = zeros(n);
  w = zeros(m);  # Used to update y.

  y = zeros(m);
  λ > 0.0 && (w2 = zeros(n));

  Anorm² = 0.0;  # Estimate of ‖A‖²_F.
  Dnorm  = 0.0;  # Estimate of ‖(A'A)⁻¹‖.
  xNorm² = 0.0;  # Estimate of ‖x‖².

  iter = 0;
  itmax == 0 && (itmax = m + n);

  rNorm  = β₁;
  rNorms = [rNorm;];
  ɛ_c = atol + rtol * rNorm;  # Stopping tolerance for consistent systems.
  ɛ_i = atol;                 # Stopping tolerance for inconsistent systems.
  verbose && @printf("%5s  %8s  %8s  %8s\n", "Aprod", "‖r‖", "‖x‖²", "‖A‖²")
  verbose && @printf("%5d  %8.2e  %8.2e  %8.2e\n", 1, rNorm, xNorm², Anorm²);

  status = "unknown";
  solved = rNorm <= ɛ_c;
  inconsistent = false;
  tired = iter >= itmax;

  while ! (solved || inconsistent || tired)
    # Generate the next Golub-Kahan vectors
    # 1. αv = A'u - βv
    BLAS.scal!(n, -β, v, 1);
    BLAS.axpy!(n, 1.0, A' * u, 1, v, 1);
    α = BLAS.nrm2(n, v, 1);
    if α == 0.0
      inconsistent = true;
      continue;
    end
    BLAS.scal!(n, 1.0/α, v, 1);

    Anorm² += α * α + λ * λ;

    if λ > 0.0
      # Givens rotation to zero out the δ in position (k, 2k):
      #      k-1  k   2k     k   2k      k-1  k   2k
      # k   [ θ   α   δ ] [ c₁   s₁ ] = [ θ   ρ      ]
      # k+1 [     β     ] [ s₁  -c₁ ]   [     θ+   γ ]
      ρ  = sqrt(α * α + δ * δ);
      c₁ =  α / ρ;
      s₁ = -δ / ρ;
    else
      ρ = α;
    end

    ξ = -θ / ρ * ξ;

    if λ > 0.0
      # w1 = c₁ * v + s₁ * w2
      # w2 = s₁ * v - c₁ * w2
      # x  = x + ξ * w1
      # Save storage on w1 since it cannot be updated
      # using a BLAS call.
      for i = 1 : n
        x[i]  = x[i] + ξ * (c₁ * v[i] + s₁ * w2[i]);
        w2[i] = s₁ * x[i] - c₁ * w2[i];
      end
    else
      BLAS.axpy!(n, ξ, v, 1, x, 1);  # x = x + ξ * v;
    end

    # Recur y.
    BLAS.scal!(m, -θ/ρ_prev, w, 1);
    BLAS.axpy!(m, 1.0, u, 1, w, 1);  # w = u - θ/ρ_prev * w;
    BLAS.axpy!(m, ξ/ρ, w, 1, y, 1);  # y = y + ξ/ρ * w;

    # 2. βu = A v - αu
    BLAS.scal!(m, -α, u, 1);
    BLAS.axpy!(m, 1.0, A * v, 1, u, 1);
    β = BLAS.nrm2(m, u, 1);
    β > 0.0 && BLAS.scal!(m, 1.0/β, u, 1);

    # Finish  updates from the first Givens rotation.
    if λ > 0.0
      θ =  β * c₁;
      γ =  β * s₁;
    else
      θ = β;
    end

    if λ > 0.0
      # Givens rotation to zero out the γ in position (k+1, 2k)
      #       2k  2k+1    2k  2k+1      2k  2k+1
      # k+1 [  γ    λ ] [ c₂   s₂ ] = [  0    δ ]
      # k+2 [  0    0 ] [ s₂  -c₂ ]   [  0    0 ]
      δ  =  sqrt(γ * γ + λ * λ);
      c₂ = -λ / δ;
      s₂ =  γ / δ;
      BLAS.scal!(n, s₂, w2, 1);
    end

    Anorm² += β * β;
    xNorm² += ξ * ξ;
    rNorm   = β * abs(ξ);          # r = -     β * ξ * u
    λ > 0.0 && (rNorm *= abs(c₁)); # r = -c₁ * β * ξ * u when λ > 0.
    push!(rNorms, rNorm);
    iter = iter + 1;

    ρ_prev = ρ;  # Only differs from α if λ > 0.

    verbose && @printf("%5d  %8.2e  %8.2e  %8.2e\n", 1 + 2 * iter, rNorm, xNorm², Anorm²);

    solved = rNorm <= ɛ_c;
    inconsistent = false;
    tired = iter >= itmax;
  end

  # TODO: transfer to LSQR point and update y.

  status = tired ? "maximum number of iterations exceeded" : (inconsistent ? "system probably inconsistent" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, inconsistent, rNorms, [], status);
  return (x, y, stats);
end
