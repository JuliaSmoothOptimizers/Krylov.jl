include("get_div_grad.jl")

cr_tol = 1.0e-6
rtol = 1.0e-6
atol = 1.0e-8
itmax = 10

# Cubic spline matrix _ case: ‖x*‖ > Δ
n = 10
A = spdiagm((ones(n-1), 4 * ones(n), ones(n-1)), (-1, 0, 1))
b = A * [1:n;]

(x, stats) = cr(A, b, atol, rtol, itmax)
r = b - A * x
resid = norm(r) / norm(b)
@printf("CR: Relative residual: %8.1e\n", resid)
@test(resid <= cr_tol)
@test(stats.solved)

# Code coverage
(x, stats) = cr(full(A), b)
show(stats)

radius = 0.75 * norm(x)
(x, stats) = cr(A, b, atol, rtol, itmax, radius)
show(stats)
@test(stats.solved)
@test_approx_eq(norm(x), radius)

# Sparse Laplacian
A = get_div_grad(16, 16, 16)
b = randn(size(A, 1))
itmax = 0
  # case: ‖x*‖ > Δ
radius = 10.
(x, stats) = cr(A, b, atol, rtol, itmax, radius)
xNorm = norm(x)
r = b - A * x
resid = norm(r) / norm(b)
@printf("CR: Relative residual: %8.1e\n", resid)
@test_approx_eq(xNorm, radius)
@test(stats.solved)
  # case: ‖x*‖ < Δ
radius = 30.
(x, stats) = cr(A, b, atol, rtol, itmax, radius)
xNorm = norm(x)
r = b - A * x
resid = norm(r) / norm(b)
@printf("CR: Relative residual: %8.1e\n", resid)
@test(resid <= cr_tol)
@test(stats.solved)

radius = 0.75 * xNorm
itmax = 10
(x, stats) = cr(A, b, atol, rtol, itmax, radius)
show(stats)
@test(stats.solved)
@test(abs(radius - norm(x)) <= cr_tol * radius)

opA = LinearOperator(A)
(xop, statsop) = cr(opA, b, atol, rtol, itmax, radius)
@test xop == x

n = 100
itmax = 2 * n
B = LBFGSOperator(n)
srand(0)
for i = 1:5
  push!(B, rand(n), rand(n))
end
b = B * ones(n)
(x, stats) = cr(B, b, atol, rtol, itmax)
@test x ≈ ones(n)
@test stats.solved

# Test b == 0
(x, stats) = cr(A, zeros(size(A,1)))
@test x == zeros(size(A,1))
@test stats.status == "x = 0 is a zero-residual solution"

# Test integer values
A = [4 -1 0; -1 4 -1; 0 -1 4]
b = [7; 2; -1]
(x, stats) = cr(A, b)
@test stats.solved
