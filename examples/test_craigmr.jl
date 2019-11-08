using Krylov, HarwellRutherfordBoeing
using LinearAlgebra, Printf

# M = HarwellBoeingMatrix("data/gemat1.rra")
M = HarwellBoeingMatrix("data/wm2.rra")
A = M.matrix
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

x_exact = A' * ones(m)
x_exact_norm = norm(x_exact)
x_exact /= x_exact_norm
b = A * x_exact
(x, y, stats) = craigmr(A, b)
show(stats)
resid = norm(A * x - b) / norm(b)
@printf("CRAIGMR: Relative residual: %7.1e\n", resid)
@printf("CRAIGMR: ‖x - x*‖₂: %7.1e\n", norm(x - x_exact))
@printf("CRAIGMR: %d iterations\n", length(stats.residuals))
