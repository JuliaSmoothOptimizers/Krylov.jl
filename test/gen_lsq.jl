# A test problem coming from LSQR.
function lstp(nrow :: Int, ncol :: Int, ndupl :: Int, npower :: Int, λ :: Real, x :: Array)

  # LSTP  generates a sparse least-squares test problem of the form
  #
  #   minimize ‖ [A ] x - [b] ‖
  #            ‖ [λI]     [0] ‖
  #
  # having a specified solution x.  The matrix A is constructed
  # in the form A = HY*D*HZ, where D is an nrow by ncol diagonal matrix,
  # and HY and HZ are Householder transformations.

  @assert(nrow ≥ ncol)

  # Construct two unit vectors for the Householder transformations.
  # fourpi = 4π
  fourpi = 4 * 3.141592  # This is the approximation used in Mike's original subroutine.
  α = fourpi / nrow  # 4π / nrow
  β = fourpi / ncol  # 4π / ncol
  hy = map(sin, [1:nrow;] * α)
  hz = map(cos, [1:ncol;] * β)

  α = norm(hy)
  hy /= α
  HY = I - 2 * hy' * hy  # HY is nrow x nrow.
  β = norm(hz)
  hz /= β
  HZ = I - 2 * hz' * hz  # HZ is ncol x ncol.

  # Set the diagonal matrix D containing the singular values of A.
  d = (div.(([0:ncol-1;] .+ ndupl), ndupl) * ndupl / ncol).^npower  # Integer div!
  D = diagm(nrow, ncol, d)
  A = HY * D * HZ

  Acond = abs(d[ncol] / d[1])

  # Compute residual vector.
  r = zeros(nrow)
  x = Float64.(x)
  r[1:ncol]  = HZ * x ./ d
  t = 1.0
  for i = ncol + 1 : nrow
    j = i - ncol
    r[i] = t * j / nrow
    t = -t
  end
  r = HY * r

  # Compute right-hand side b = r + Ax.
  rnorm = norm(r)
  b = r + A * x

  return (b, A, D, HY, HZ, Acond, rnorm)
end


function test(nrow, ncol, ndupl, npower, damp)

  x = ncol .- [1:ncol;]  # Desired solution.
  return lstp(nrow, ncol, ndupl, npower, damp, x)
end


function testall()
  damp = 0  # Must be zero for this problem to be consistent.
  test(40, 40, 4, 1, damp)
  test(40, 40, 4, 2, damp)
  test(40, 40, 4, 3, damp)
  test(40, 40, 4, 4, damp)
end
