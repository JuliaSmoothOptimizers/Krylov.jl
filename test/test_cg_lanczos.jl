function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

@testset "cg_lanczos" begin
  cg_tol = 1.0e-6
  n = 10

  # Cubic splines matrix.
  A, b = symmetric_definite()
  b_norm = norm(b)

  (x, stats) = cg_lanczos(A, b, itmax=n)
  resid = norm(b - A * x) / b_norm
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  # Test negative curvature detection.
  A[n-1,n-1] = -4.0
  (x, stats) = cg_lanczos(A, b, check_curvature=true)
  @test(stats.status == "negative curvature")
  A[n-1,n-1] = 4.0

  shifts=Float64.([1:6;])

  (x, stats) = cg_lanczos(A, b, shifts, itmax=n)
  r = residuals(A, b, shifts, x)
  resids = map(norm, r) / b_norm
  @test(all(resids .≤ cg_tol))
  @test(stats.solved)

  # Test negative curvature detection.
  shifts = [-4.0; -3.0; 2.0]
  (x, stats) = cg_lanczos(A, b, shifts, check_curvature=true, itmax=n)
  @test(stats.flagged == [true, true, false])

  # Code coverage.
  (x, stats) = cg_lanczos(Matrix(A), b)
  (x, stats) = cg_lanczos(Matrix(A), b, Float64.([1:6;]))

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = cg_lanczos(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"
  (x, stats) = cg_lanczos(A, zeros(size(A,1)), Float64.([1:6;]))
  @test x == [zeros(size(A,1)) for i=1:6]
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = cg_lanczos(A, b, M=M)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ cg_tol)
  @test(stats.solved)

  shifts = Float64.([1:10;])
  (x, stats) = cg_lanczos(A, b, shifts, M=M)
  r = residuals(A, b, shifts, x)
  resids = map(norm, r) / norm(b)
  @test(all(resids .≤ cg_tol))
  @test(stats.solved)
end
