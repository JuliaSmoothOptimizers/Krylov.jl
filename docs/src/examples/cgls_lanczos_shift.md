```@example cgls_lanczos_shift
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ A' * (A * x - b) + shifts[i] * x[i] for i = 1 : nshifts ]
  return r
end
ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "well1033")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define regularization parameters.
shifts = [1.0, 2.0, 3.0, 4.0]

(x, stats) = cgls_lanczos_shift(A, b, shifts)
show(stats)
r = residuals(A, b, shifts, x)

resids = map(norm, r) / norm(b)
@printf("CGLS: Relative residuals with shifts:\n")
for resid in resids
  @printf(" %8.1e", resid)
end
@printf("\n")
```
