include("get_div_grad.jl")

cr_tol = 1.0e-6
Δ = 10.
itmax = 10

# Cubic spline matrix
n = 10
A = spdiagm((ones(n-1), 4 * ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:n;]

(x, stats) = cr(A, b, Δ, cr_tol, itmax)
r = b - A * x
resid = norm(r) / norm(b)
@printf("CR: Relative residual: %8.1e\n", resid)
# @test(resid <= cr_tol)
@test(stats.solved)

# Code coverage
(x, stats) = cr(A, b)
show(stats)

Δ = 0.75 * norm(x)
(x, stats) = cr(A, b, Δ, cr_tol, itmax)
show(stats)
@test(stats.solved)
@test(abs(Δ - norm(x)) <= cr_tol * Δ)

# Sparse Laplacian.
A = get_div_grad(16, 16, 16)
b = randn(size(A, 1))
(x, stats) = cr(A, b)
r = b - A * x
resid = norm(r) / norm(b)
@printf("CR: Relative residual: %8.1e\n", resid)
# @test(resid <= cr_tol)
@test(stats.solved)

Δ = 0.75 * norm(x)
(x, stats) = cr(A, b, Δ, cr_tol, itmax)
show(stats)
@test(stats.solved)
@test(abs(Δ - norm(x)) <= cr_tol * Δ)

opA = LinearOperator(A)
(xop, statsop) = cr(opA, b, Δ, cr_tol, itmax)
@test xop == x

Δ = 10.
n = 100
itmax = 2 * n
B = LBFGSOperator(n)
srand(0)
for i = 1:5
  push!(B, rand(n), rand(n))
end
b = B * ones(n)
(x, stats) = cr(B, b, Δ, cr_tol, itmax)
@test x ≈ ones(n)
@test stats.solved

# Test b == 0
(x, stats) = cr(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [4 -1 0; -1 4 -1; 0 -1 4]
A = LinearOperator(A)
b = [7; 2; -1]
(x, stats) = cr(A, b)
@test stats.solved
