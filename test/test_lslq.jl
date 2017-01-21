lslq_tol = 1.0e-5

for npower = 1 : 4
  (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b)
  resid = norm(A' * (A*x - b)) / norm(b)
  @printf("LSLQ: Relative residual: %8.1e\n", resid)
  @test(resid <= lslq_tol)
  @test(stats.solved)

  λ = 1.0e-3
  (x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, b, λ=λ)
  resid = norm(A' * (A*x - b) + λ * λ * x) / norm(b)
  @printf("LSLQ: Relative residual: %8.1e\n", resid)
  @test(resid <= lslq_tol)
  @test(stats.solved)
end

# Code coverage.
(b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, 3, 0)
(x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(full(A), b)
(x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(sparse(full(A)), b)
show(stats)

# Test b == 0
(x, x_cg, err_lbnds, err_ubnds_lq, err_ubnds_cg, stats) = lslq(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [eye(Int, 3); rand(1:10, 2, 3)]
b = A * ones(Int, 3)
@test norm(lslq(A, b)[1] - ones(3)) < 1e-11
