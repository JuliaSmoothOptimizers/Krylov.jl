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

# Test error estimate
A = full(get_div_grad(8, 8, 8))
b = ones(size(A, 1))
λest = (1-1e-10)*eigmin(A)
x_exact = A\b
(x, xcg, stats) = symmlq(A, b, λest=λest)
err = norm(x_exact - x)
errcg = norm(x_exact - xcg)
@printf("SYMMLQ    : true error: %8.1e\n", err)
@printf("SYMMLQ-CG : true error: %8.1e\n", errcg)
@printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
@printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
@test( err <= stats.errors[end] )
@test( errcg <= stats.errorscg[end])
(x, xcg, stats) = symmlq(A, b, λest=λest, window=1)
@printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
@printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
@test( err <= stats.errors[end] )
@test( errcg <= stats.errorscg[end])
(x, xcg, stats) = symmlq(A, b, λest=λest, window=5)
@printf("SYMMLQ    : err up-bnd : %8.1e\n", stats.errors[end])
@printf("SYMMLQ-CG : err up-bnd : %8.1e\n", stats.errorscg[end])
@test( err <= stats.errors[end] )
@test( errcg <= stats.errorscg[end] )
