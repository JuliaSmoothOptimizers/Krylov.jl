crls_tol = 1.0e-5;

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

  (x, stats) = crls(A, b);
  resid = norm(A' * (A*x - b)) / norm(b)
  @printf("CRLS: Relative residual: %8.1e\n", resid);
  @test(resid <= crls_tol);
  @test(stats.solved);

  λ = 1.0e-3;
  (x, stats) = crls(A, b, λ=λ);
  resid = norm(A' * (A*x - b) + λ * x) / norm(b)
  @printf("CRLS: Relative residual: %8.1e\n", resid);
  @test(resid <= crls_tol);
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

(x, stats) = crls(A, b, M=M);
resid = norm(A' * M * (A * x - b)) / sqrt(dot(b, M * b));
@printf("CRLS: Preconditioned residual: %8.1e\n", resid);
@test resid <= crls_tol;

# Code coverage.
(b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
(x, stats) = crls(full(A), b);
(x, stats) = crls(sparse(full(A)), b);
show(stats);

