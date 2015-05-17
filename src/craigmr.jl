# An implementation of CRAIG-MR for the solution of the
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
# This method is equivalent to CRMR, and is described in
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
# Montreal, QC, May 2015.

export craigmr

# Methods for various argument types.
include("craigmr_methods.jl")

@doc """
Solve the consistent linear system

  Ax + √λs = b

using the CRAIG-MR method, where λ ≥ 0 is a regularization parameter.
This method is equivalent to applying the Conjugate Residuals method
to the normal equations of the second kind

  (AA' + λI) y = b

but is more stable. When λ = 0, this method solves the minimum-norm problem

  min ‖x‖₂  s.t.  x ∈ argmin ‖Ax - b‖₂.

When λ > 0, this method solves the problem

  min ‖(x,s)‖₂  s.t. Ax + √λs = b.

CRAIGMR produces monotonic residuals ‖r‖₂.
It is formally equivalent to CRMR, though can be slightly more accurate,
and intricate to implement. Both the x- and y-parts of the solution are
returned.
""" ->
function craigmr(A :: LinearOperator, b :: Array{Float64,1};
                 λ :: Float64=0.0, atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                 itmax :: Int=0, verbose :: Bool=false)

  m, n = size(A);
  size(b, 1) == m || error("Inconsistent problem size");
  verbose && @printf("CRAIG-MR: system of %d equations in %d variables\n", m, n);

  # Compute y such that AA'y = b. Then recover x = A'y.
  y = zeros(m);
  β₁ = BLAS.nrm2(m, b, 1);   # Marginally faster than norm(b);
  if β₁ == 0.0
    x = zeros(n);
    return (x, y, SimpleStats(true, false, [0.0], [], "x = 0 is a zero-residual solution"));
  end
  β = β₁;

  # Initialize Golub-Kahan process.
  # β₁ u₁ = b.
  u = copy(b);
  BLAS.scal!(m, 1.0/β₁, u, 1);
  v = copy(A' * u);
  α = BLAS.nrm2(n, v, 1);
  Anorm² = α * α;

  verbose && @printf("%5s  %7s  %7s  %7s  %7s  %8s  %8s  %7s\n",
                     "Aprod", "‖r‖", "‖A'r‖", "β", "α", "cos", "sin", "‖A‖²");
  verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n",
                     1, β₁, α, β₁, α, 0, 1, Anorm²);

  if α == 0.0
    # A'b = 0 so x = 0 is a minimum least-squares solution
    x = zeros(n);
    return (x, y, SimpleStats(true, false, [β₁], [0.0], "x = 0 is a minimum least-squares solution"));
  end
  BLAS.scal!(n, 1.0/α, v, 1);
 
  # Initialize other constants.
  ζbar = β₁;
  ρbar = α;
  θ = 0.0;
  rNorm = ζbar;
  rNorms = [rNorm];
  ArNorm = α;
  ArNorms = [ArNorm];

  ɛ_c = atol + rtol * rNorm;   # Stopping tolerance for consistent systems.
  ɛ_i = atol + rtol * ArNorm;  # Stopping tolerance for inconsistent systems.

  iter = 0;
  itmax == 0 && (itmax = m + n);

  wbar = copy(u);
  BLAS.scal!(m, 1.0/α, wbar, 1);
  w = zeros(m);

  status = "unknown";
  solved = rNorm <= ɛ_c
  inconsistent = (rNorm > 1.0e+2 * ɛ_c) & (ArNorm <= ɛ_i)
  tired  = iter >= itmax

  while ! (solved || inconsistent || tired)
    iter = iter + 1;

    # Generate next Golub-Kahan vectors.
    # 1. βu = Av - αu
    BLAS.scal!(m, -α, u, 1);
    BLAS.axpy!(m, 1.0, A * v, 1, u, 1);
    β = norm(u);
    β != 0.0 && BLAS.scal!(m, 1.0/β, u, 1);
    Anorm² = Anorm² + β * β;  # = ‖B_{k-1}‖²

    # Continue QR factorization
    #
    # Q [ Lₖ  β₁ e₁ ] = [ Rₖ   zₖ  ] :
    #   [ β    0    ]   [ 0   ζbar ]
    #
    #       k  k+1    k    k+1      k  k+1
    # k   [ c   s ] [ ρbar    ] = [ ρ  θ⁺    ]
    # k+1 [ s  -c ] [ β    α⁺ ]   [    ρbar⁺ ]
    #
    # so that we obtain
    #
    # [ c  s ] [ ζbar ] = [ ζ     ]
    # [ s -c ] [  0   ]   [ ζbar⁺ ]
    (c, s, ρ) = sym_givens(ρbar, β);
    ζ = c * ζbar;
    ζbar = s * ζbar;
    rNorm = abs(ζbar);
    push!(rNorms, rNorm);

    BLAS.scal!(m, -θ/ρ, w, 1);
    BLAS.axpy!(m, 1.0/ρ, wbar, 1, w, 1);  # w = (wbar - θ * w) / ρ;
    BLAS.axpy!(m, ζ, w, 1, y, 1);         # y = y + ζ * w;

    # 2. αv = A'u - βv
    BLAS.scal!(n, -β, v, 1);
    BLAS.axpy!(n, 1.0, A' * u, 1, v, 1);
    α = BLAS.nrm2(n, v, 1);
    Anorm² = Anorm² + α * α;  # = ‖Lₖ‖
    ArNorm = α * β * abs(ζ/ρ);
    push!(ArNorms, ArNorm);

    verbose && @printf("%5d  %7.1e  %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %7.1e\n",
                       1 + 2 * iter, rNorm, ArNorm, β, α, c, s, Anorm²);

    if α != 0.0
      BLAS.scal!(n, 1.0/α, v, 1);
      BLAS.scal!(m, -β/α, wbar, 1);
      BLAS.axpy!(m, 1.0/α, u, 1, wbar, 1);  # wbar = (u - beta * wbar) / alpha;
    end
    θ = s * α;
    ρbar = -c * α;

    solved = rNorm <= ɛ_c
    inconsistent = (rNorm > 1.0e+2 * ɛ_c) & (ArNorm <= ɛ_i)
    tired  = iter >= itmax
  end

  x = A' * y;

  status = tired ? "maximum number of iterations exceeded" : (solved ? "found approximate minimum-norm solution" : "found approximate minimum least-squares solution")
  stats = SimpleStats(solved, inconsistent, rNorms, ArNorms, status);
  return (x, y, stats)
end
