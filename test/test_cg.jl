include("get_div_grad.jl")

cg_tol = 1.0e-6;

# Cubic spline matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:n];

for mat in {A, full(A), LinearOperator(A)}
  (x, stats) = cg(mat, b, itmax=10);
  show(stats);
  r = b - A * x;
  resid = norm(r) / norm(b)
  @printf("CG: Relative residual: %8.1e\n", resid);
  @test(resid <= cg_tol);
  @test(stats.solved);
end

# Sparse Laplacian.
A = get_div_grad(16, 16, 16);
b = randn(size(A, 1));

(x, stats) = cg(A, b);
show(stats);
r = b - A * x;
resid = norm(r) / norm(b);
@printf("CG: Relative residual: %8.1e\n", resid);
@test(resid <= cg_tol);
@test(stats.solved);

