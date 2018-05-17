include("get_div_grad.jl")

symmlq_tol = 1.0e-5

# 1. Symmetric and positive definite systems.
#
# Cubic spline matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1)); b = A * [1:n;]

(x, xcg, stats) = symmlq(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("SYMMLQ: Relative residual: %8.1e\n", resid)
@test(resid <= symmlq_tol)
@test(stats.solved)

# Symmetric indefinite variant.
A = A - 3 * speye(n)
b = A * [1:n;]
(x, xcg, stats) = symmlq(A, b, itmax=n+1)
r = b - A * x
resid = norm(r) / norm(b)
@printf("SYMMLQ: Relative residual: %8.1e\n", resid)
@test(resid <= symmlq_tol)
@test(stats.solved)

# Code coverage.
(x, xcg, stats) = symmlq(full(A), b)
show(stats)

# Sparse Laplacian (CG point will terminate sooner).
A = get_div_grad(16, 16, 16)
b = ones(size(A, 1))
(x, xcg, stats) = symmlq(A, b, atol=1e-12, rtol=1e-12)
r = b - A * x
resid = norm(r) / norm(b)
@show stats
@printf("SYMMLQ: Relative residual: %8.1e\n", resid)
@test(resid <= symmlq_tol)
@test(stats.solved)

# Symmetric indefinite variant, almost singular.
A = A - 5 * speye(size(A, 1))
(x, xcg, stats) = symmlq(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("SYMMLQ: Relative residual: %8.1e\n", resid)
@test(resid <= 100 * symmlq_tol)
@test(stats.solved)

# Test b == 0
(x, xcg, stats) = symmlq(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = spdiagm((ones(Int, n-1), 4*ones(Int, n), ones(Int, n-1)), (-1, 0, 1)); b = A * [1:n;]
(x, xcg, stats) = symmlq(A, b)
@test stats.solved
