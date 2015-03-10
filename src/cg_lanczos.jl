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


function cg_lanczos{T <: Real}(A :: LinearOperator, b :: Array{T,1};
                               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                               verbose :: Bool=false)

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

  # Define stopping tolerance.
  rNorm = σ;
  ε = atol + rtol * rNorm;
  verbose && @printf("%5d  %8.1e\n", iter, rNorm);

  # Main loop.
  while (rNorm > ε) & (iter < itmax)
    # Form next Lanczos vector.
    v_next = A * v;
    # δ = BLAS.dot(n, v, 1, v_next, 1); doesn't seem to pay off.
    δ = dot(v, v_next);
    BLAS.axpy!(n, -δ, v, 1, v_next, 1);  # Faster than v_next = Av - δ * v;
    if iter > 0
      BLAS.axpy!(n, -β, v_prev, 1, v_next, 1);  # Faster than v_next = v_next - β * v_prev;
      v_prev = v;
    end
    β = norm(v_next);
    v = v_next / β;

    # Compute next CG iterate.
    γ = 1 / (δ - ω / γ);
    #     x = x + γ * p;
    BLAS.axpy!(n, γ, p, 1, x, 1);  # Faster than x = x + γ * p;

    ω = β * γ;
    σ = -ω * σ;
    ω = ω * ω;
    BLAS.scal!(n, ω, p, 1);
    BLAS.axpy!(n, σ, v, 1, p, 1);  # Faster than p = σ * v + ω * p;
    rNorm = abs(σ);
    iter = iter + 1;
    verbose && @printf("%5d  %8.1e\n", iter, rNorm);
  end
  return x;
end


function cg_lanczos_shift_seq{Tb <: Real, Ts <: Real}(A :: LinearOperator, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                      atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                      verbose :: Bool=false)

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

  # Keep track of shifted systems that have converged.
  converged = falses(nshifts);
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  # Initialize some constants used in recursions below.
  σ = β * ones(nshifts);
  δhat = zeros(nshifts);
  ω = zeros(nshifts);
  γ = ones(nshifts);

  # Define stopping tolerance.
  rNorms = β * ones(nshifts);
  ε = atol + rtol * β;

  # Build format strings for printing.
  if verbose
    fmt = "%5d";
    for i = 1 : nshifts
      fmt = fmt * "  %8.1e";
    end
    fmt = fmt * "\n";
    print_formatted(fmt, iter, rNorms...);
  end

  # Main loop.
  while ! all(converged) & (iter < itmax)
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

    # Compute next CG iterate for each shift.
    not_cv = find(! converged);

    #     δhat[not_cv] = δ + shifts[not_cv];
    #     γ[not_cv] = 1 ./ (δhat[not_cv] - ω[not_cv] ./ γ[not_cv]);
    #     x[:, not_cv] += (p[:, not_cv] * diagm(γ[not_cv]));  # diagm?!
    #     ω[not_cv] = β * γ[not_cv];
    #     σ[not_cv] .*= -ω[not_cv];
    #     ω[not_cv] .*= ω[not_cv];
    #     p[:, not_cv] = (v * σ[not_cv]' + p[:, not_cv] * diagm(ω[not_cv]));
    #     rNorms[not_cv] = abs(σ[not_cv]);
    #     converged[not_cv] = rNorms[not_cv] .<= ε;

    # Loop is a bit faster than the vectorized version.
    for i in not_cv
      δhat[i] = δ + shifts[i];
      γ[i] = 1 ./ (δhat[i] - ω[i] ./ γ[i]);
      x[:, i] += γ[i] * p[:, i];

      ω[i] = β * γ[i];
      σ[i] *= -ω[i];
      ω[i] *= ω[i];
      p[:, i] = v * σ[i] + p[:, i] * ω[i];

      # Update list of systems that have converged.
      rNorms[i] = abs(σ[i]);
      converged[i] = rNorms[i] <= ε;
    end

    iter = iter + 1;
    verbose && print_formatted(fmt, iter, rNorms...);
  end
  return x;
end


function cg_lanczos_shift_par{Tb <: Real, Ts <: Real}(A :: LinearOperator, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                                                      atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                                                      verbose :: Bool=false)

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
  dconverged = dfill(false, (nshifts,), workers(), [nchunks]);
  iter = 0;
  itmax == 0 && (itmax = 2 * n);

  # Initialize some constants used in recursions below.
  dσ = dfill(β, (nshifts,), workers(), [nchunks]);
  dδ = dzeros((nshifts,), workers(), [nchunks]);
  dω = dzeros((nshifts,), workers(), [nchunks]);
  dγ = dones((nshifts,), workers(), [nchunks]);

  # Define stopping tolerance.
  drNorm = dfill(β, (nshifts,), workers(), [nchunks]);
  ε = atol + rtol * β;

  # Build format strings for printing.
  if verbose
    fmt = "%5d";
    for i = 1 : nshifts
      fmt = fmt * "  %8.1e";
    end
    fmt = fmt * "\n";
    print_formatted(fmt, iter, drNorm...);
  end

  # Main loop.
  while ! preduce(&, dconverged) & (iter < itmax)
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
        converged_loc = localpart(dconverged);
        not_cv = find(! converged_loc);

        if ! all(converged_loc)

          # Fetch parts of relevant arrays for which
          # the residual has not yet converged.
          σ_loc = localpart(dσ);
          δ_loc = localpart(dδ);
          γ_loc = localpart(dγ);
          ω_loc = localpart(dω);
          x_loc = localpart(dx);
          p_loc = localpart(dp);
          rNorm_loc = localpart(drNorm);
          shifts_loc = localpart(dshifts); shifts_loc = shifts_loc[not_cv];

          δ_loc[not_cv] = δ + shifts_loc;
          γ_loc[not_cv] = 1 ./ (δ_loc[not_cv] - ω_loc[not_cv] ./ γ_loc[not_cv]);
          x_loc[:, not_cv] += (p_loc[:, not_cv] * diagm(γ_loc[not_cv]));  # diagm?!

          ω_loc[not_cv] = β * γ_loc[not_cv];
          σ_loc[not_cv] .*= -ω_loc[not_cv];
          ω_loc[not_cv] .*= ω_loc[not_cv];
          p_loc[:, not_cv] = (v * σ_loc[not_cv]' + p_loc[:, not_cv] * diagm(ω_loc[not_cv]));
          rNorm_loc[not_cv] = abs(σ_loc[not_cv]);

          # Update list of systems that have converged.
          converged_loc[not_cv] = rNorm_loc[not_cv] .<= ε;
        end
      end
    end

    iter = iter + 1;
    verbose && print_formatted(fmt, iter, drNorm...);
  end
  return convert(Array, dx);
end
