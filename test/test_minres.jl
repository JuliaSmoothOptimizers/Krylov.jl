include("get_div_grad.jl")

minres_tol = 1.0e-6

# 1. Symmetric and positive definite systems.
#
# Cubic spline matrix.
n = 10;
A = spdiagm((ones(n-1), 4*ones(n), ones(n-1)), (-1, 0, 1)); b = A * [1:n;]

(x, stats) = minres(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("MINRES: Relative residual: %8.1e\n", resid)
@test(resid <= minres_tol)
@test(stats.solved)

# radius = 0.75 * norm(x)
# (x, stats) = minres(A, b, radius=radius, itmax=10)
# show(stats)
# @test(stats.solved)
# @test(abs(radius - norm(x)) <= minres_tol * radius)

# Symmetric indefinite variant.
A = A - 3 * speye(n)
b = A * [1:n;]
(x, stats) = minres(A, b, itmax=10)
r = b - A * x
resid = norm(r) / norm(b)
@printf("MINRES: Relative residual: %8.1e\n", resid)
@test(resid <= minres_tol)
@test(stats.solved)

# Code coverage.
(x, stats) = minres(full(A), b)
show(stats)

# Sparse Laplacian.
A = get_div_grad(16, 16, 16)
b = ones(size(A, 1))
(x, stats) = minres(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("MINRES: Relative residual: %8.1e\n", resid)
@test(resid <= minres_tol)
@test(stats.solved)

# radius = 0.75 * norm(x)
# (x, stats) = minres(A, b, radius=radius, itmax=10)
# show(stats)
# @test(stats.solved)
# @test(abs(radius - norm(x)) <= minres_tol * radius)

# Symmetric indefinite variant, almost singular.
A = A - 5 * speye(size(A, 1))
(x, stats) = minres(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("MINRES: Relative residual: %8.1e\n", resid)
@test(resid <= 100 * minres_tol)
@test(stats.solved)

# Test b == 0
(x, stats) = minres(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = spdiagm((ones(Int, n-1), 4*ones(Int, n), ones(Int, n-1)), (-1, 0, 1)); b = A * [1:n;]
(x, stats) = minres(A, b)
@test stats.solved
