using Random

include("get_div_grad.jl")

function test_cg()
  cg_tol = 1.0e-6;

  # Cubic spline matrix.
  n = 10;
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * [1:n;];

  (x, stats) = cg(A, b, itmax=10);
  r = b - A * x;
  resid = norm(r) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid);
  @test(resid <= cg_tol);
  @test(stats.solved);

  # Code coverage.
  (x, stats) = cg(Matrix(A), b);
  show(stats);

  radius = 0.75 * norm(x);
  (x, stats) = cg(A, b, radius=radius, itmax=10);
  show(stats)
  @test(stats.solved);
  @test(abs(radius - norm(x)) <= cg_tol * radius);

  # Sparse Laplacian.
  A = get_div_grad(16, 16, 16);
  b = randn(size(A, 1));
  (x, stats) = cg(A, b);
  r = b - A * x;
  resid = norm(r) / norm(b);
  @printf("CG: Relative residual: %8.1e\n", resid);
  @test(resid <= cg_tol);
  @test(stats.solved);

  radius = 0.75 * norm(x);
  (x, stats) = cg(A, b, radius=radius, itmax=10);
  show(stats)
  @test(stats.solved);
  @test(abs(radius - norm(x)) <= cg_tol * radius);

  opA = LinearOperator(A)
  (xop, statsop) = cg(opA, b, radius=radius, itmax=10)
  @test(abs(radius - norm(xop)) <= cg_tol * radius)

  n = 100
  B = LBFGSOperator(n)
  Random.seed!(0)
  for i = 1:5
    push!(B, rand(n), rand(n))
  end
  b = B * ones(n)
  (x, stats) = cg(B, b, itmax=2n)
  @test norm(x - ones(n)) ≤ cg_tol * norm(x)
  @test stats.solved

  # Test b == 0
  (x, stats) = cg(A, zeros(size(A,1)))
  @test x == zeros(size(A,1))
  @test stats.status == "x = 0 is a zero-residual solution"

  # Test integer values
  A = [4 -1 0; -1 4 -1; 0 -1 4]
  b = [7; 2; -1]
  (x, stats) = cg(A, b)
  @test stats.solved

  # Test with Jacobi (or diagonal) preconditioner
  A = ones(10,10) + 9 * I
  b = 10 * [1:10;]
  M = 1/10 * opEye(10)
  (x, stats) = cg(A, b, M=M, itmax=10);
  show(stats)
  r = b - A * x;
  resid = norm(r) / norm(b);
  @printf("CG: Relative residual: %8.1e\n", resid);
  @test(resid <= cg_tol);
  @test(stats.solved);
end

test_cg()
