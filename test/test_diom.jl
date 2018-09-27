include("get_div_grad.jl")

diom_tol = 1.0e-6

# 1. Symmetric and positive definite systems.
#
# Cubic spline matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1)); b = A * [1:n;]

(x, stats) = diom(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("DIOM: Relative residual: %8.1e\n", resid)
@test(resid <= diom_tol)
@test(stats.solved)

# Symmetric indefinite variant.
A = A - 3 * speye(n)
b = A * [1:n;]
(x, stats) = diom(A, b, itmax=10)
r = b - A * x
resid = norm(r) / norm(b)
@printf("DIOM: Relative residual: %8.1e\n", resid)
@test(resid <= diom_tol)
@test(stats.solved)

# Code coverage.
(x, stats) = diom(full(A), b)
show(stats)

# Sparse Laplacian.
A = get_div_grad(16, 16, 16)
b = ones(size(A, 1))
(x, stats) = diom(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("DIOM: Relative residual: %8.1e\n", resid)
@test(resid <= diom_tol)
@test(stats.solved)

# Symmetric indefinite variant, almost singular.
A = A - 5 * speye(size(A, 1))
(x, stats) = diom(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("DIOM: Relative residual: %8.1e\n", resid)
@test(resid <= 100 * diom_tol)
@test(stats.solved)

# Test b == 0
(x, stats) = diom(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = spdiagm((ones(Int, n-1), 4*ones(Int, n), ones(Int, n-1)), (-1, 0, 1)); b = A * [1:n;]
(x, stats) = diom(A, b)
@test stats.solved

# Test with Jacobi (or diagonal) preconditioner
A = ones(10,10) + 9 * eye(10)
b = 10 * [1:10;]
M = 1/10 * opEye(10)
(x, stats) = diom(A, b, M=M)
show(stats)
r = b - A * x
resid = norm(r) / norm(b)
@printf("DIOM: Relative residual: %8.1e\n", resid)
@test(resid <= diom_tol)
@test(stats.solved)
