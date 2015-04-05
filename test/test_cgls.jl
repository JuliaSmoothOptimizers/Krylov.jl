cgls_tol = 1.0e-6;

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

# Code coverage.
(b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
(x, stats) = cgls(full(A), b);
(x, stats) = cgls(sparse(full(A)), b);
show(stats);

