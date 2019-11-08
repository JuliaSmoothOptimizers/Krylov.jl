using DistributedArrays, MatrixMarket
using Krylov, LinearOperators
using Printf, LinearAlgebra, SparseArrays, Distributed

function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1)
  r = [ (b - A * x[:,i] - shifts[i] * x[:,i]) for i = 1 : nshifts ]
  return r
end

# Parallel reduce.
preduce(func, darray) = reduce(func, map(fetch, [ (@spawnat p reduce(func, localpart(darray))) for p = procs(darray) ] ))

"""The Lanczos version of the conjugate gradient method to solve a family
of shifted systems

  (A + αI) x = b  (α = α₁, α₂, ...)

The method does _not_ abort if A + αI is not definite.

In this version, the shifted systems are dispatched on the available processors,
and operations specific to each shift is carried out on the processor hosting it.
"""
function cg_lanczos_shift_par(A :: AbstractLinearOperator, b :: Array{Tb,1}, shifts :: Array{Ts,1};
                              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                              check_curvature :: Bool=false, verbose :: Bool=false) where {Tb, Ts <: Real}

  n = size(b, 1)
  (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size")

  dshifts = distribute(shifts)
  nshifts = size(shifts, 1)
  nchunks = length(dshifts.indices)
  verbose && @printf("CG Lanczos: system of %d equations in %d variables with %d shifts\n", n, n, nshifts)

  # Initial state.
  ## Distribute x similarly to shifts.
  dx = dzeros((n, nshifts), workers(), [1, nchunks])
  β = norm(b)
  β == 0 && return convert(Array, dx),
            Krylov.LanczosStats(true, zeros(nshifts), false, 0.0, 0.0, "x = 0
                                is a zero-residual solution")
  v = b / β
  v_prev = v

  # Distribute p similarly to shifts.
  # Initialize each p to b.
  dp = dzeros((n, nshifts), workers(), [1, nchunks])
  @sync for proc in procs(dp)
    @spawnat proc begin
      p_loc = localpart(dp)
      for i = 1 : size(p_loc, 2)
        p_loc[:,i] = b
      end
    end
  end

  # Keep track of shifted systems that have converged.
  dsolved = dfill(false, (nshifts,), workers(), [nchunks])
  dconverged = dfill(false, (nshifts,), workers(), [nchunks])
  dindefinite= dfill(false, (nshifts,), workers(), [nchunks])
  iter = 0
  itmax == 0 && (itmax = 2 * n)

  # Initialize some constants used in recursions below.
  dσ = dfill(β, (nshifts,), workers(), [nchunks])
  dδ = dzeros((nshifts,), workers(), [nchunks])
  dω = dzeros((nshifts,), workers(), [nchunks])
  dγ = dones((nshifts,), workers(), [nchunks])

  # Define stopping tolerance.
  drNorm = dfill(β, (nshifts,), workers(), [nchunks])
  ε = atol + rtol * β

  # Build format strings for printing.
  if verbose
    fmt = "%5d" * repeat("  %8.1e", nshifts) * "\n"
    c_printf(fmt, iter, drNorm...)
  end

  solved = false
  tired = iter >= itmax
  status = "unknown"

  # Main loop.
  while ! (solved || tired)
    # Form next Lanczos vector.
    v_next = A * v
    δ = dot(v, v_next)
    BLAS.axpy!(n, -δ, v, 1, v_next, 1) # Faster than v_next = v_next - δ * v
    if iter > 0
      BLAS.axpy!(n, -β, v_prev, 1, v_next, 1) # Faster than v_next = v_next - β * v_prev
      v_prev = v
    end
    β = norm(v_next)
    v = v_next / β

    # Compute next CG iterate for each shift.
    @sync for proc in procs(dp)
      @spawnat proc begin
        solved_loc = localpart(dsolved)
        converged_loc = localpart(dconverged)
        indefinite_loc = localpart(dindefinite)
        shifts_loc = localpart(dshifts)
        δ_loc = localpart(dδ)
        γ_loc = localpart(dγ)
        ω_loc = localpart(dω)

        rNorm_loc = localpart(drNorm)

        # Check curvature: v'(A + sᵢI)v = v'Av + sᵢ ‖v‖² = δ + sᵢ because ‖v‖ = 1.
        # It is possible to show that σⱼ² (δⱼ + sᵢ - ωⱼ₋₁ / γⱼ₋₁) = pⱼᵀ (A + sᵢ I) pⱼ.
        # Stop iterating on indefinite problems if requested.
        δ_loc[:] = δ .+ shifts_loc
        γ_loc[:] = 1 ./ (δ_loc - ω_loc ./ γ_loc)
        indefinite_loc[:] .|= (γ_loc .<= 0.0)
        not_cv = check_curvature ? findall(.! (converged_loc .| indefinite_loc)) : findall(.! converged_loc)

        if length(not_cv) > 0
          # Fetch parts of relevant arrays for which the residual has not yet converged.
          σ_loc = localpart(dσ)
          x_loc = localpart(dx)
          p_loc = localpart(dp)
          shifts_loc = shifts_loc[not_cv]

          ω_loc[not_cv] = β * γ_loc[not_cv]
          σ_loc[not_cv] .*= -ω_loc[not_cv]
          ω_loc[not_cv] .*= ω_loc[not_cv]
          rNorm_loc[not_cv] = abs.(σ_loc[not_cv])

          # Can't seem to use BLAS3 GEMM calls here.
          x_loc[:, not_cv] += (p_loc[:, not_cv] * diagm(0 => γ_loc[not_cv]))
          p_loc[:, not_cv] = (v * σ_loc[not_cv]' + p_loc[:, not_cv] * diagm(0 => ω_loc[not_cv]))

          # Update list of systems that have converged.
          converged_loc[not_cv] = rNorm_loc[not_cv] .<= ε
        end

        solved_loc[:] = converged_loc .| indefinite_loc
      end
    end

    iter = iter + 1;
    verbose && c_printf(fmt, iter, drNorm...)

    solved = preduce(&, dsolved)
    tired = iter >= itmax
  end

  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"
  stats = Krylov.LanczosStats(solved, convert(Array, drNorm), convert(Array, dindefinite), 0.0, 0.0, status)
  return (dx, stats)
end

# mtx = "data/bcsstk09.mtx"
mtx = "data/1138bus.mtx"

A = MatrixMarket.mmread(mtx)
n = size(A, 1)
b = ones(n)
b_norm = norm(b)

# Define a linear operator with preallocation.
op = PreallocatedLinearOperator(A, symmetric=true)

# Solve (A + αI)x = b in parallel.
shifts = [1.0, 2.0, 3.0, 4.0]
(x, stats) = cg_lanczos_shift_par(op, b, shifts, verbose=false)
r = residuals(A, b, shifts, convert(Array, x))
resids = map(norm, r) / b_norm
@printf("Relative residuals with shifts:\n")
for resid in resids
  @printf(" %8.1e", resid)
end
@printf("\n")
