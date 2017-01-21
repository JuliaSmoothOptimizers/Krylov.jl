craigmr_tol = 1.0e-6;

function test_craigmr(A, b; λ=0.0)
  (nrow, ncol) = size(A);
  (x, y, stats) = craigmr(A, b, λ=λ);
  r = b - A * x;
  Ar = A' * r;
  # if λ > 0
  #   s = r / sqrt(λ);
  #   r = r - sqrt(λ) * s;
  # end
  resid = norm(r) / norm(b);
  Aresid = norm(Ar) / (norm(A) * norm(b));
  @printf("CRAIGMR: residual: %7.1e  least-squares: %7.1e\n", resid, Aresid);
  return (x, y, stats, resid, Aresid)
end

# Underdetermined consistent.
A = rand(10, 25); b = A * ones(25);
(x, y, stats, resid, Aresid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(resid <= craigmr_tol);
@test(stats.solved);
(xI, xmin, xmin_norm) = check_min_norm(A, b, x);
@test(norm(xI - xmin) <= cond(A) * craigmr_tol * xmin_norm);

# Underdetermined inconsistent.
A = ones(10, 25); b = rand(10); b[1] = -1.0;
(x, y, stats, resid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(stats.inconsistent);
@test(stats.Aresiduals[end] <= craigmr_tol);

# Square consistent.
A = rand(10, 10); b = A * ones(10);
(x, y, stats, resid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(resid <= craigmr_tol);
@test(stats.solved);
(xI, xmin, xmin_norm) = check_min_norm(A, b, x);
@test(norm(xI - xmin) <= cond(A) * craigmr_tol * xmin_norm);

# Square inconsistent.
A = ones(10, 10); b = rand(10); b[1] = -1.0;
(x, y, stats, resid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(stats.inconsistent);
@test(stats.Aresiduals[end] <= craigmr_tol);

# Overdetermined consistent.
A = rand(25, 10); b = A * ones(10);
(x, y, stats, resid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(resid <= craigmr_tol);
@test(stats.solved);
(xI, xmin, xmin_norm) = check_min_norm(A, b, x);
@test(norm(xI - xmin) <= cond(A) * craigmr_tol * xmin_norm);

# Overdetermined inconsistent.
A = ones(5, 3); b = rand(5); b[1] = -1.0;
(x, y, stats, resid) = test_craigmr(A, b);
@test(norm(x - A' * y) <= craigmr_tol * norm(x));
@test(stats.inconsistent);
@test(stats.Aresiduals[end] <= craigmr_tol);

# With regularization, all systems are underdetermined and consistent.
# (x, y, stats, resid) = test_craigmr(A, b, λ=1.0e-3);
# @test(norm(x - A' * y) <= craigmr_tol * norm(x));
# @test(resid <= craigmr_tol);
# @test(stats.solved);
# (xI, xmin, xmin_norm) = check_min_norm(A, b, x, λ=1.0e-3);
# @test(norm(xI - xmin) <= cond(A) * craigmr_tol * xmin_norm);

# Code coverage.
(x, y, stats) = craigmr(sparse(A), b, λ=1.0e-3);
show(stats);

# Test b == 0
(x, y, stats) = craigmr(A, zeros(size(A,1)), λ=1.0e-3)
@test x == zeros(size(A,2))
@test y == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [eye(Int, 3); rand(1:10, 2, 3)]
b = A * ones(Int, 3)
(x, y, stats) = craigmr(A, b)
@test stats.solved
