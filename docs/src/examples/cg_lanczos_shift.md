```@example cg_lanczos_shift
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

function residuals(A, b, shifts, x)
  nshifts = length(shifts)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end
ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "1138_bus")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
n, m = size(A)
b = ones(n)

# Solve (A + αI)x = b.
shifts = [1.0, 2.0, 3.0, 4.0]
(x, stats) = cg_lanczos_shift(A, b, shifts)
show(stats)
r = residuals(A, b, shifts, x)
resids = map(norm, r) / norm(b)
@printf("Relative residuals with shifts:\n")
for resid in resids
  @printf(" %8.1e", resid)
end
@printf("\n")
```
