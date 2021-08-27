@testset "diom" begin
  diom_tol = 1.0e-6

  # Symmetric and positive definite system.
  A, b = symmetric_definite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Symmetric indefinite variant.
  A, b = symmetric_indefinite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Nonsymmetric and positive definite systems.
  A, b = nonsymmetric_definite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Nonsymmetric indefinite variant.
  A, b = nonsymmetric_indefinite()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Code coverage.
  (x, stats) = diom(sparse(A), b)

  # Sparse Laplacian.
  A, b = sparse_laplacian()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Symmetric indefinite variant, almost singular.
  A, b = almost_singular()
  (x, stats) = diom(A, b)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Test b == 0
  A, b = zero_rhs()
  (x, stats) = diom(A, b)
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Poisson equation in polar coordinates.
  A, b = polar_poisson()
  (x, stats) = diom(A, b, memory=200)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Test restart
  A, b = restart()
  solver = DiomSolver(A, b, 20)
  diom!(solver, A, b, itmax=50)
  @test !solver.stats.solved
  diom!(solver, A, b, restart=true)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test solver.stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A, b, M = square_preconditioned()
  (x, stats) = diom(A, b, M=M)
  r = b - A * x
  resid = norm(M * r) / norm(M * b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Right preconditioning
  A, b, N = square_preconditioned()
  (x, stats) = diom(A, b, N=N)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)

  # Split preconditioning
  A, b, M, N = two_preconditioners()
  (x, stats) = diom(A, b, M=M, N=N)
  r = b - A * x
  resid = norm(M * r) / norm(M * b)
  @test(resid ≤ diom_tol)
  @test(stats.solved)
end
