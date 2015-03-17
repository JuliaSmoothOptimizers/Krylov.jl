include("get_div_grad.jl")

cg_tol = 1.0e-6;

# Cubic spline matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:10];

x = cg(A, b, itmax=10);
r = b - A * x;
resid = norm(r) / norm(b)
@printf("CG: Relative residual: %8.1e\n", resid);
@test(resid <= cg_tol);

# Sparse Laplacian.
A = get_div_grad(32, 32, 32);
b = randn(size(A, 1));
x = cg(A, b);
r = b - A * x;
resid = norm(r) / norm(b);
@printf("CG: Relative residual: %8.1e\n", resid);
@test(resid <= cg_tol);
