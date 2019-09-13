using HarwellRutherfordBoeing
using Krylov, LinearOperators
using LinearAlgebra, Printf

# M = HarwellBoeingMatrix("data/gemat1.rra")
M = HarwellBoeingMatrix("data/wm2.rra")
A = M.matrix
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a linear operator with preallocation.
Ap = zeros(m)
Aᵀq = zeros(n)
op = LinearOperator(m, n, false, false,
                    p -> mul!(Ap, A, p),
                    q -> mul!(Aᵀq, transpose(A), q),
                    q -> mul!(Aᵀq, transpose(A), q))

x_exact = A' * ones(m) # y = ones(m)
x_exact_norm = norm(x_exact)
x_exact /= x_exact_norm
b = A * x_exact
(x, stats) = crmr(op, b)
show(stats)
resid = norm(A * x - b) / norm(b)
@printf("CRMR: Relative residual: %7.1e\n", resid)
@printf("CRMR: ‖x - x*‖₂: %7.1e\n", norm(x - x_exact))
