## CG

```@example cg
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "bcsstk09")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = ones(n)
b_norm = norm(b)

# Solve Ax = b.
(x, stats) = cg(A, b)
show(stats)
r = b - A * x
@printf("Relative residual: %8.1e\n", norm(r) / b_norm)
```

## CG-LANCZOS

```@example cg_lanczos
using Krylov, MatrixMarket, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

function residuals(A, b, shifts, x)
  nshifts = size(shifts, 1)
  r = [ (b - A * x[i] - shifts[i] * x[i]) for i = 1 : nshifts ]
  return r
end
ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "1138_bus")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = ones(n)
b_norm = norm(b)

# Solve Ax = b.
(x, stats) = cg_lanczos(A, b)
show(stats)
r = b - A * x
@printf("Relative residual without shift: %8.1e\n", norm(r) / norm(b))

# Solve (A + αI)x = b sequentially.
shifts = [1.0, 2.0, 3.0, 4.0]
(x, stats) = cg_lanczos(A, b, shifts)
show(stats)
r = residuals(A, b, shifts, x)
resids = map(norm, r) / b_norm
@printf("Relative residuals with shifts:\n")
for resid in resids
  @printf(" %8.1e", resid)
end
@printf("\n")
```

## SYMMLQ

```@example symmlq
using Krylov
using LinearAlgebra, Printf

A = diagm([1.0; 2.0; 3.0; 0.0])
n = size(A, 1)
b = [1.0; 2.0; 3.0; 0.0]
b_norm = norm(b)

# SYMMLQ returns the minimum-norm solution of symmetric, singular and consistent systems
(x, stats) = symmlq(A, b, transfer_to_cg=false);
r = b - A * x;

@printf("Residual r: %s\n", Krylov.vec2str(r))
@printf("Relative residual norm ‖r‖: %8.1e\n", norm(r) / b_norm)
@printf("Solution x: %s\n", Krylov.vec2str(x))
@printf("Minimum-norm solution? %s\n", x ≈ [1.0; 1.0; 1.0; 0.0])
```

## MINRES-QLP

```@example minres_qlp
using Krylov
using LinearAlgebra, Printf

A = diagm([1.0; 2.0; 3.0; 0.0])
n = size(A, 1)
b = [1.0; 2.0; 3.0; 4.0]
b_norm = norm(b)

# MINRES-QLP returns the minimum-norm solution of symmetric, singular and inconsistent systems
(x, stats) = minres_qlp(A, b);
r = b - A * x;

@printf("Residual r: %s\n", Krylov.vec2str(r))
@printf("Relative residual norm ‖r‖: %8.1e\n", norm(r) / b_norm)
@printf("Solution x: %s\n", Krylov.vec2str(x))
@printf("Minimum-norm solution? %s\n", x ≈ [1.0; 1.0; 1.0; 0.0])
```

## TriCG

```@example tricg
using Krylov, LinearOperators
using LinearAlgebra, Printf, SparseArrays

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Symmetric quasi-definite systems and variants
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
M = diagm(0 => [3.0 * i for i = 1:n])
N = diagm(0 => [5.0 * i for i = 1:n])
c = -b

# [I   A] [x] = [b]
# [Aᵀ -I] [y]   [c]
(x, y, stats) = tricg(A, b, c)
K = [eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [-I   A] [x] = [b]
# [ Aᵀ  I] [y]   [c]
(x, y, stats) = tricg(A, b, c, flip=true)
K = [-eye(m) A; A' eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [I   A] [x] = [b]
# [Aᵀ  I] [y]   [c]
(x, y, stats) = tricg(A, b, c, spd=true)
K = [eye(m) A; A' eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [-I    A] [x] = [b]
# [ Aᵀ  -I] [y]   [c]
(x, y, stats) = tricg(A, b, c, snd=true)
K = [-eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [τI    A] [x] = [b]
# [ Aᵀ  νI] [y]   [c]
(τ, ν) = (1e-4, 1e2)
(x, y, stats) = tricg(A, b, c, τ=τ, ν=ν)
K = [τ*eye(m) A; A' ν*eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriCG: Relative residual: %8.1e\n", resid)

# [M⁻¹  A  ] [x] = [b]
# [Aᵀ  -N⁻¹] [y]   [c]
(x, y, stats) = tricg(A, b, c, M=M, N=N, verbose=1)
K = [inv(M) A; A' -inv(N)]
H = BlockDiagonalOperator(M, N)
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, H * r))
@printf("TriCG: Relative residual: %8.1e\n", resid)
```

## TriMR

```@example trimr
using Krylov, LinearOperators, LDLFactorizations
using LinearAlgebra, Printf, SparseArrays

# Identity matrix.
eye(n::Int) = sparse(1.0 * I, n, n)

# Saddle-point systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
D = diagm(0 => [2.0 * i for i = 1:n])
m, n = size(A)
c = -b

# [D   A] [x] = [b]
# [Aᵀ  0] [y]   [c]
llt_D = cholesky(D)
opD⁻¹ = LinearOperator(Float64, 5, 5, true, true, (y, v) -> ldiv!(y, llt_D, v))
opH⁻¹ = BlockDiagonalOperator(opD⁻¹, eye(n))
(x, y, stats) = trimr(A, b, c, M=opD⁻¹, sp=true)
K = [D A; A' zeros(n,n)]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)

# Symmetric quasi-definite systems
n = m = 5
A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
b = ones(n)
M = diagm(0 => [3.0 * i for i = 1:n])
N = diagm(0 => [5.0 * i for i = 1:n])
c = -b

# [I   A] [x] = [b]
# [Aᵀ -I] [y]   [c]
(x, y, stats) = trimr(A, b, c)
K = [eye(m) A; A' -eye(n)]
B = [b; c]
r = B - K * [x; y]
resid = norm(r)
@printf("TriMR: Relative residual: %8.1e\n", resid)

# [M   A] [x] = [b]
# [Aᵀ -N] [y]   [c]
ldlt_M = ldl(M)
ldlt_N = ldl(N)
opM⁻¹ = LinearOperator(Float64, size(M,1), size(M,2), true, true, (y, v) -> ldiv!(y, ldlt_M, v))
opN⁻¹ = LinearOperator(Float64, size(N,1), size(N,2), true, true, (y, v) -> ldiv!(y, ldlt_N, v))
opH⁻¹ = BlockDiagonalOperator(opM⁻¹, opN⁻¹)
(x, y, stats) = trimr(A, b, c, M=opM⁻¹, N=opN⁻¹, verbose=1)
K = [M A; A' -N]
B = [b; c]
r = B - K * [x; y]
resid = sqrt(dot(r, opH⁻¹ * r))
@printf("TriMR: Relative residual: %8.1e\n", resid)
```

## BICGSTAB

```@example bicgstab
using Krylov, LinearOperators, IncompleteLU, HarwellRutherfordBoeing
using LinearAlgebra, Printf, SuiteSparseMatrixCollection, SparseArrays

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "sherman5")
path = fetch_ssmc(matrix, format="RB")

n = matrix.nrows[1]
A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
b = A * ones(n)

F = ilu(A, τ = 0.05)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with BICGSTAB and an incomplete LU factorization
# Remark: CGS can be used in the same way
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))

# Without preconditioning
x, stats = bicgstab(A, b, history=true)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = bicgstab(A, b, history=true, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = bicgstab(A, b, history=true, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = bicgstab(A, b, history=true, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
```

## DQGMRES

```@example dqgmres
using Krylov, LinearOperators, ILUZero, MatrixMarket
using LinearAlgebra, Printf, SuiteSparseMatrixCollection

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "Simon", "raefsky1")
path = fetch_ssmc(matrix, format="MM")

n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = A * ones(n)

F = ilu0(A)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with DQGMRES and an ILU(0) preconditioner
# Remark: DIOM, FOM and GMRES can be used in the same way
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))

# Without preconditioning
x, stats = dqgmres(A, b, memory=50, history=true)
r = b - A * x
@printf("[Without preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Without preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Split preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, M=opM, N=opN)
r = b - A * x
@printf("[Split preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Split preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Left preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, M=opP)
r = b - A * x
@printf("[Left preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Left preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)

# Right preconditioning
x, stats = dqgmres(A, b, memory=50, history=true, N=opP)
r = b - A * x
@printf("[Right preconditioning] Residual norm: %8.1e\n", norm(r))
@printf("[Right preconditioning] Number of iterations: %3d\n", length(stats.residuals) - 1)
```

## CGNE

```@example cgne
using Krylov, HarwellRutherfordBoeing, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "wm2")
path = fetch_ssmc(matrix, format="RB")

A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

x_exact = A' * ones(m)
x_exact_norm = norm(x_exact)
x_exact /= x_exact_norm
b = A * x_exact
(x, stats) = cgne(A, b)
show(stats)
resid = norm(A * x - b) / norm(b)
@printf("CGNE: Relative residual: %7.1e\n", resid)
@printf("CGNE: ‖x - x*‖₂: %7.1e\n", norm(x - x_exact))
```

## CRMR

```@example crmr
using Krylov, HarwellRutherfordBoeing, SuiteSparseMatrixCollection
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "gemat1")
path = fetch_ssmc(matrix, format="RB")

A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

x_exact = A' * ones(m)
x_exact_norm = norm(x_exact)
x_exact /= x_exact_norm
b = A * x_exact
(x, stats) = crmr(A, b)
show(stats)
resid = norm(A * x - b) / norm(b)
@printf("CRMR: Relative residual: %7.1e\n", resid)
@printf("CRMR: ‖x - x*‖₂: %7.1e\n", norm(x - x_exact))
```

## CRAIG

```@example craig
using Krylov
using LinearAlgebra, Printf

m = 5
n = 8
λ = 1.0e-3
A = rand(m, n)
b = A * ones(n)
xy_exact = [A  λ*I] \ b # In Julia, this is the min-norm solution!

(x, y, stats) = craig(A, b, λ=λ, atol=0.0, rtol=1.0e-20, verbose=1)
show(stats)

# Check that we have a minimum-norm solution.
# When λ > 0 we solve min ‖(x,s)‖  s.t. Ax + λs = b, and we get s = λy.
@printf("Primal feasibility: %7.1e\n", norm(b - A * x - λ^2 * y) / norm(b))
@printf("Dual   feasibility: %7.1e\n", norm(x - A' * y) / norm(x))
@printf("Error in x: %7.1e\n", norm(x - xy_exact[1:n]) / norm(xy_exact[1:n]))
if λ > 0.0
  @printf("Error in y: %7.1e\n", norm(λ * y - xy_exact[n+1:n+m]) / norm(xy_exact[n+1:n+m]))
end
```

## CRAIGMR

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

## CGLS

```@example cgls
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "well1033")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter.
λ = 1.0e-3

(x, stats) = cgls(A, b, λ=λ)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("CGLS: Relative residual: %8.1e\n", resid)
@printf("CGLS: ‖x‖: %8.1e\n", norm(x))
```

## CRLS

```@example crls
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "well1850")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter.
λ = 1.0e-3

(x, stats) = crls(A, b, λ=λ)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("CRLS: Relative residual: %8.1e\n", resid)
@printf("CRLS: ‖x‖: %8.1e\n", norm(x))
```

## LSQR

```@example lsqr
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "illc1033")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter and a preconditioner.
λ = 1.0e-3
λ > 0.0 && (N = I / λ)

(x, stats) = lsqr(A, b, λ=λ, sqd=λ > 0, atol=0.0, btol=0.0, N=N)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("LSQR: Relative residual: %8.1e\n", resid)
@printf("LSQR: ‖x‖: %8.1e\n", norm(x))
```

## LSMR

```@example lsmr
using MatrixMarket, SuiteSparseMatrixCollection
using Krylov, LinearOperators
using LinearAlgebra, Printf

ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "HB", "illc1850")
path = fetch_ssmc(matrix, format="MM")

A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
b = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1])_b.mtx"))[:]
(m, n) = size(A)
@printf("System size: %d rows and %d columns\n", m, n)

# Define a regularization parameter and a preconditioner.
λ = 1.0e-3
λ > 0.0 && (N = I / λ)

(x, stats) = lsmr(A, b, λ=λ, sqd=λ > 0, atol=0.0, btol=0.0, N=N)
show(stats)
resid = norm(A' * (A * x - b) + λ * x) / norm(b)
@printf("LSMR: Relative residual: %8.1e\n", resid)
@printf("LSMR: ‖x‖: %8.1e\n", norm(x))
```
