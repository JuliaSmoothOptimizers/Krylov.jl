```@example craigmr
using Krylov, HarwellRutherfordBoeing, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "wm1")
path = fetch_ssmc(matrix, format="RB")

A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
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
```
