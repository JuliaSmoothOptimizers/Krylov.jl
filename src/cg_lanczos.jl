# An implementation of the Lanczos version of the conjugate gradient method
# for a family of shifted systems of the form (A + αI) x = b.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips
# Regularization, SIAM Journal on Scientific Computing, 20(5),
# pp. 1831-1850, 1999.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cg_lanczos, cg_lanczos_shift_seq, cg_lanczos_shift_par

# Methods for various argument types.
include("cg_lanczos_methods.jl")

@doc """
The Lanczos version of the conjugate gradient method to solve the
symmetric linear system

  Ax = b

The method does _not_ abort if A is not definite.
""" ->
function cg_lanczos{T <: Real}(A :: LinearOperator, b :: Array{T,1};
                               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                               check_curvature :: Bool=false, verbose :: Bool=false)

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");

  # Initial state.
  x = zeros(n);
  β = norm(b);
  β == 0 && return x;
  v = b / β;
  v_prev = v;
  p = copy(b);
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  # Initialize some constants used in recursions below.
  σ = β;
  ω = 0;
  γ = 1;
  Anorm2 = 0.0;
  β_prev = 0.0;

  # Define stopping tolerance.
  rNorm = σ;
  rNorms = [rNorm;];
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e\n", iter, rNorm);

  indefinite = false;
  solved = rNorm <= ε;
  tired = iter >= itmax;
  status = "unknown";

  # Main loop.
  while ! (solved || tired || (check_curvature & indefinite))
    # Form next Lanczos vector.
    v_next = A * v;
    δ = dot(v, v_next);  # BLAS.dot(n, v, 1, v_next, 1) doesn't seem to pay off.

    # Check curvature. Exit fast if requested.
    indefinite |= (δ <= 0.0);
    (check_curvature & indefinite) && continue;

    BLAS.axpy!(n, -δ, v, 1, v_next, 1);  # Faster than v_next = Av - δ * v;
    if iter > 0
      BLAS.axpy!(n, -β, v_prev, 1, v_next, 1);  # Faster than v_next = v_next - β * v_prev;
      v_prev = v;
    end
    β = norm(v_next);
    v = v_next / β;
    Anorm2 += β_prev^2 + β^2 + δ^2;  # Use ‖T‖ as increasing approximation of ‖A‖.
    β_prev = β;

    # Compute next CG iterate.
    γ = 1 / (δ - ω / γ);
    BLAS.axpy!(n, γ, p, 1, x, 1);  # Faster than x = x + γ * p;

    ω = β * γ;
    σ = -ω * σ;
    ω = ω * ω;
    BLAS.scal!(n, ω, p, 1);
    BLAS.axpy!(n, σ, v, 1, p, 1);  # Faster than p = σ * v + ω * p;
    rNorm = abs(σ);
    push!(rNorms, rNorm);
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e\n", iter, rNorm);
    solved = rNorm <= ε;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : (check_curvature & indefinite) ? "negative curvature" : "solution good enough given atol and rtol"
  stats = LanczosStats(solved, rNorms, indefinite, sqrt(Anorm2), 0.0, status);  # TODO: Estimate Acond.
  return (x, stats);
end


@doc """
The Lanczos version of the conjugate gradient method to solve a family
of shifted systems

  (A + αI) x = b  (α = α₁, α₂, ...)

The method does _not_ abort if A + αI is not definite.
""" ->
function cg_lanczos_shift_seq{Tb <: Real, Ts <: Real}(A :: LinearOperator, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                      atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                      check_curvature :: Bool=false, verbose :: Bool=false)

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");

  nshifts = size(shifts, 1);

  # Initial state.
  ## Distribute x similarly to shifts.
  x = zeros(n, nshifts);
  β = norm(b);
  β == 0 && return x;
  v = b / β;
  v_prev = copy(v);

  # Initialize each p to b.
  p = b * ones(nshifts)';

  # Initialize some constants used in recursions below.
  σ = β * ones(nshifts);
  δhat = zeros(nshifts);
  ω = zeros(nshifts);
  γ = ones(nshifts);

  # Define stopping tolerance.
  rNorms = β * ones(nshifts);
  rNorms_history = [rNorms;];
  ε = atol + rtol * β;

  # Keep track of shifted systems that have converged.
  converged = rNorms .<= ε;
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  # Keep track of shifted systems with negative curvature if required.
  indefinite = falses(nshifts);

  # Build format strings for printing.
  if verbose
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n";
    c_printf(fmt, iter, rNorms...);
  end

  solved = all(converged);
  tired = iter >= itmax;
  status = "unknown";

  # Main loop.
  while ! (solved || tired)
    # Form next Lanczos vector.
    v_next = A * v;
    δ = dot(v, v_next);
    BLAS.axpy!(n, -δ, v, 1, v_next, 1);  # Faster than v_next = Av - δ * v;
    if iter > 0
      BLAS.axpy!(n, -β, v_prev, 1, v_next, 1);  # Faster than v_next = v_next - β * v_prev;
      v_prev = v;
    end
    β = norm(v_next);
    v = v_next / β;

    # Check curvature: v'(A + sᵢI)v = v'Av + sᵢ ‖v‖² = δ + sᵢ because ‖v‖ = 1.
    indefinite |= (δ + shifts .<= 0.0);

    # Compute next CG iterate for each shifted system that has not yet converged.
    # Stop iterating on indefinite problems if requested.
    not_cv = check_curvature ? find(! (converged | indefinite)) : find(! converged);

    # Loop is a bit faster than the vectorized version.
    for i in not_cv
      δhat[i] = δ + shifts[i];

      γ[i] = 1 ./ (δhat[i] - ω[i] ./ γ[i]);
      x[:, i] += γ[i] * p[:, i];  # Strangely, this is faster than a loop.

      ω[i] = β * γ[i];
      σ[i] *= -ω[i];
      ω[i] *= ω[i];
      p[:, i] = v * σ[i] + p[:, i] * ω[i];  # Faster than loop.

      # Update list of systems that have converged.
      rNorms[i] = abs(σ[i]);
      converged[i] = rNorms[i] <= ε;
    end

    # Is there a better way than to update this array twice per iteration?
    not_cv = check_curvature ? find(! (converged | indefinite)) : find(! converged);

    length(not_cv) > 0 && append!(rNorms_history, rNorms);
    iter = iter + 1;
    verbose && c_printf(fmt, iter, rNorms...);

    solved = length(not_cv) == 0;
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = LanczosStats(solved, reshape(rNorms_history, nshifts, int(sum(size(rNorms_history))/nshifts))', indefinite, 0.0, 0.0, status);  # TODO: Estimate Anorm and Acond.
  return (x, stats);
end


@doc """
The Lanczos version of the conjugate gradient method to solve a family
of shifted systems

  (A + αI) x = b  (α = α₁, α₂, ...)

The method does _not_ abort if A + αI is not definite.

In this version, the shifted systems are dispatched on the available processors,
and operations specific to each shift is carried out on the processor hosting it.
""" ->
function cg_lanczos_shift_par{Tb <: Real, Ts <: Real}(A :: LinearOperator, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                      atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                      check_curvature :: Bool=false, verbose :: Bool=false)

  n = size(b, 1);
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");

  dshifts = distribute(shifts);
  nshifts = size(shifts, 1);
  nchunks = size(dshifts.chunks, 1);

  # Initial state.
  ## Distribute x similarly to shifts.
  dx = dzeros((n, nshifts), workers(), [1, nchunks]);
  β = norm(b);
  β == 0 && return convert(Array, dx);
  v = b / β;
  v_prev = v;

  # Distribute p similarly to shifts.
  # Initialize each p to b.
  dp = dzeros((n, nshifts), workers(), [1, nchunks]);
  @sync for proc in procs(dp)
    @spawnat proc begin
      p_loc = localpart(dp);
      for i = 1 : size(p_loc, 2)
        p_loc[:,i] = b;
      end
    end
  end

  # Keep track of shifted systems that have converged.
  dsolved = dfill(false, (nshifts,), workers(), [nchunks]);
  dconverged = dfill(false, (nshifts,), workers(), [nchunks]);
  dindefinite= dfill(false, (nshifts,), workers(), [nchunks]);
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  # Initialize some constants used in recursions below.
  dσ = dfill(β, (nshifts,), workers(), [nchunks]);
  dδ = dzeros((nshifts,), workers(), [nchunks]);
  dω = dzeros((nshifts,), workers(), [nchunks]);
  dγ = dones((nshifts,), workers(), [nchunks]);

  # Define stopping tolerance.
  drNorm = dfill(β, (nshifts,), workers(), [nchunks]);
  #   drNorms = dfill(β, (nshifts,), workers(), [nchunks]);
  ε = atol + rtol * β;

  # Build format strings for printing.
  if verbose
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n";
    c_printf(fmt, iter, drNorm...);
  end

  solved = false;
  tired = iter >= itmax;
  status = "unknown";

  # Main loop.
  while ! (solved || tired)
    # Form next Lanczos vector.
    v_next = A * v;
    δ = dot(v, v_next);
    BLAS.axpy!(n, -δ, v, 1, v_next, 1);  # Faster than v_next = v_next - δ * v;
    if iter > 0
      BLAS.axpy!(n, -β, v_prev, 1, v_next, 1);  # Faster than v_next = v_next - β * v_prev;
      v_prev = v;
    end
    β = norm(v_next);
    v = v_next / β;

    # Compute next CG iterate for each shift.
    @sync for proc in procs(dp)
      @spawnat proc begin
        solved_loc = localpart(dsolved);
        converged_loc = localpart(dconverged);
        indefinite_loc = localpart(dindefinite);
        shifts_loc = localpart(dshifts);

        rNorm_loc = localpart(drNorm);
        #         rNorms_loc = localpart(drNorms);

        # Check curvature: v'(A + sᵢI)v = v'Av + sᵢ ‖v‖² = δ + sᵢ because ‖v‖ = 1.
        # Stop iterating on indefinite problems if requested.
        indefinite_loc[:] |= (δ + shifts_loc .<= 0.0);
        not_cv = check_curvature ? find(! (converged_loc | indefinite_loc)) : find(! converged_loc);

        if length(not_cv) > 0
          # Fetch parts of relevant arrays for which the residual has not yet converged.
          σ_loc = localpart(dσ);
          δ_loc = localpart(dδ);
          γ_loc = localpart(dγ);
          ω_loc = localpart(dω);
          x_loc = localpart(dx);
          p_loc = localpart(dp);
          shifts_loc = shifts_loc[not_cv];

          δ_loc[not_cv] = δ + shifts_loc;
          γ_loc[not_cv] = 1 ./ (δ_loc[not_cv] - ω_loc[not_cv] ./ γ_loc[not_cv]);
          ω_loc[not_cv] = β * γ_loc[not_cv];
          σ_loc[not_cv] .*= -ω_loc[not_cv];
          ω_loc[not_cv] .*= ω_loc[not_cv];
          rNorm_loc[not_cv] = abs(σ_loc[not_cv]);

          # Can't seem to use BLAS3 GEMM calls here.
          x_loc[:, not_cv] += (p_loc[:, not_cv] * diagm(γ_loc[not_cv]));  # diagm?!
          p_loc[:, not_cv] = (v * σ_loc[not_cv]' + p_loc[:, not_cv] * diagm(ω_loc[not_cv]));

          # Update list of systems that have converged.
          converged_loc[not_cv] = rNorm_loc[not_cv] .<= ε;
        end

        solved_loc[:] = converged_loc | indefinite_loc;
      end
    end

    iter = iter + 1;
    verbose && c_printf(fmt, iter, drNorm...);

    solved = preduce(&, dsolved);
    tired = iter >= itmax;
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = LanczosStats(solved, drNorm, dindefinite, 0.0, 0.0, status);
  return (dx, stats);
end
