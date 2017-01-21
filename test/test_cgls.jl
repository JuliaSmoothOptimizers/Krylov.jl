cgls_tol = 1.0e-5;

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

  (x, stats) = cgls(A, b);
  resid = norm(A' * (A*x - b)) / norm(b)
  @printf("CGLS: Relative residual: %8.1e\n", resid);
  @test(resid <= cgls_tol);
  @test(stats.solved);

  λ = 1.0e-3;
  (x, stats) = cgls(A, b, λ=λ);
  resid = norm(A' * (A*x - b) + λ * x) / norm(b)
  @printf("CGLS: Relative residual: %8.1e\n", resid);
  @test(resid <= cgls_tol);
  @test(stats.solved);
end

# Test with preconditioning.
A = rand(10, 6); b = rand(10);
M = InverseLBFGSOperator(10, 4);
for _ = 1 : 6
  s = rand(10);
  y = rand(10);
  push!(M, s, y);
end

(x, stats) = cgls(A, b, M=M);
resid = norm(A' * M * (A * x - b)) / sqrt(dot(b, M * b));
@printf("CGLS: Preconditioned residual: %8.1e\n", resid);
@test resid <= cgls_tol;

# Code coverage.
(b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
(x, stats) = cgls(full(A), b);
(x, stats) = cgls(sparse(full(A)), b);
show(stats);

# Test b == 0
(x, stats) = cgls(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [eye(Int, 3); rand(1:10, 2, 3)]
b = A * ones(Int, 3)
(x, stats) = cgls(A, b)
@test stats.solved
