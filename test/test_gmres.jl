include("get_div_grad.jl")

function test_gmres()
  gmres_tol = 1.0e-6

  # Symmetric and positive definite systems (cubic spline matrix).
  n = 10
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * [1:n;]
  (x, stats) = gmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A = A - 3 * I
  b = A * [1:n;]
  (x, stats) = gmres(A, b, itmax=10)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  (x, stats) = gmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A = [i == j ? 10*(-1.0)^(i*j) : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  (x, stats) = gmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = gmres(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A = get_div_grad(16, 16, 16)
  b = ones(size(A, 1))
  (x, stats) = gmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Test with an user-defined x₀ (restarted GMRES)
  x0 = ones(size(A, 1))
  (x, stats) = gmres(A, b, x0=x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  # Also test reorthogonalization.
  A = A - 5 * I
  (x, stats) = gmres(A, b, reorthogonalization=true)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Test b == 0
  (x, stats) = gmres(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A = spdiagm(-1 => ones(Int, n-1), 0 => 4*ones(Int, n), 1 => ones(Int, n-1))
  b = A * [1:n;]
  (x, stats) = gmres(A, b)
  @test stats.solved

  # Left preconditioning
  A = ones(10,10) + 9 * I
  b = 10 * [1:10;]
  M = 1/10 * opEye(10)
  (x, stats) = gmres(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Right preconditioning
  A = ones(10,10) + 9 * I
  b = 10 * [1:10;]
  N = 1/10 * opEye(10)
  (x, stats) = gmres(A, b, N=N)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
  @test(stats.solved)

  # Split preconditioning
  A = ones(10,10) + 9 * I
  b = 10 * [1:10;]
  M = 1/√10 * opEye(10)
  N = 1/√10 * opEye(10)
  (x, stats) = gmres(A, b, M=M, N=N)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("GMRES: Relative residual: %8.1e\n", resid)
  @test(resid ≤ gmres_tol)
@test(stats.solved)
end

test_gmres()
