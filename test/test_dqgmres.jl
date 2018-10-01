include("get_div_grad.jl")

function test_dqgmres()
  dqgmres_tol = 1.0e-6

  # 1. Symmetric and positive definite systems.
  #
  # Cubic spline matrix.
  n = 10;
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * [1:n;]

  (x, stats) = dqgmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DQGMRES: Relative residual: %8.1e\n", resid)
  @test(resid <= dqgmres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A = A - 3 * I
  b = A * [1:n;]
  (x, stats) = dqgmres(A, b, itmax=10)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DQGMRES: Relative residual: %8.1e\n", resid)
  @test(resid <= dqgmres_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = dqgmres(Matrix(A), b)
  show(stats)

  # Sparse Laplacian.
  A = get_div_grad(16, 16, 16)
  b = ones(size(A, 1))
  (x, stats) = dqgmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DQGMRES: Relative residual: %8.1e\n", resid)
  @test(resid <= dqgmres_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A = A - 5 * I
  (x, stats) = dqgmres(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DQGMRES: Relative residual: %8.1e\n", resid)
  @test(resid <= 100 * dqgmres_tol)
  @test(stats.solved)

  # Test b == 0
  (x, stats) = dqgmres(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A = spdiagm(-1 => ones(Int, n-1), 0 => 4*ones(Int, n), 1 => ones(Int, n-1))
  b = A * [1:n;]
  (x, stats) = dqgmres(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A = ones(10,10) + 9 * I
  b = 10 * [1:10;]
  M = 1/10 * opEye(10)
  (x, stats) = dqgmres(A, b, M=M)
  show(stats)
  r = b - A * x
  resid = norm(r) / norm(b)
  @printf("DQGMRES: Relative residual: %8.1e\n", resid)
  @test(resid <= dqgmres_tol)
  @test(stats.solved)
end

test_dqgmres()
