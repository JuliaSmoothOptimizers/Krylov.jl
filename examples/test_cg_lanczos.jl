using Krylov, MatrixMarket
using LinearAlgebra, Printf

function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end

# mtx = "data/bcsstk09.mtx"
mtx = "data/1138bus.mtx"

A = MatrixMarket.mmread(mtx)
n = size(A, 1)
b = ones(n)
b_norm = norm(b)

# Solve Ax = b.
(x, stats) = cg_lanczos(A, b)
show(stats)
r = b - A * x
@printf("Relative residual without shift: %8.1e\n", norm(r) / norm(b))

# Solve (A + αI)x = b sequentially.
shifts = [1.0, 2.0, 3.0, 4.0]
(x, stats) = cg_lanczos_shift_seq(A, b, shifts, verbose=false)
show(stats)
r = residuals(A, b, shifts, x)
resids = map(norm, r) / b_norm
@printf("Relative residuals with shifts:\n")
for resid in resids
  @printf(" %8.1e", resid)
end
@printf("\n")
