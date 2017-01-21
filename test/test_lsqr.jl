lsqr_tol = 1.0e-5;

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0);  # No regularization.

  (x, stats) = lsqr(A, b);
  resid = norm(A' * (A*x - b)) / norm(b)
  @printf("LSQR: Relative residual: %8.1e\n", resid);
  @test(resid <= lsqr_tol);
  @test(stats.solved);

  λ = 1.0e-3;
  (x, stats) = lsqr(A, b, λ=λ);
  resid = norm(A' * (A*x - b) + λ * λ * x) / norm(b)
  @printf("LSQR: Relative residual: %8.1e\n", resid);
  @test(resid <= lsqr_tol);
  @test(stats.solved);
end

# Code coverage.
(b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0);
(x, stats) = lsqr(full(A), b);
(x, stats) = lsqr(sparse(full(A)), b);
show(stats);

# Test b == 0
(x, stats) = lsqr(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [eye(Int, 3); rand(1:10, 2, 3)]
b = A * ones(Int, 3)
(x, stats) = lsqr(A, b)
@test stats.solved
