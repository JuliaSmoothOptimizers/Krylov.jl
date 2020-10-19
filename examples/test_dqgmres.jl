using Krylov, LinearOperators, ILU0
using LinearAlgebra, Printf

n = 64
N = n^3
A = spdiagm(
      -1 => fill(-1.0, n - 1), 
       0 => fill(3.0, n), 
       1 => fill(-2.0, n - 1)
    )
Id = sparse(1.0I, n, n)
A = kron(A, Id) + kron(Id, A)
A = kron(A, Id) + kron(Id, A)
x = ones(N)
b = A * x
F = ilu0(A)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with DQGMRES and an ILU(0) preconditioner
yM = zeros(N)
yN = zeros(N)
yP = zeros(N)
opM = LinearOperator(Float64, N, N, false, false, y -> forward_substitution(yM, F, y))
opN = LinearOperator(Float64, N, N, false, false, y -> backward_substitution(yN, F, y))
opP = LinearOperator(Float64, N, N, false, false, y -> ldiv!(yP, F, y))

# Split preconditioning
x, stats = dqgmres(A, b, M=opM, N=opN, verbose=true)
show(stats)
r = b - A * x
@printf("Residual norm: %8.1e\n", norm(r))

# Left preconditioning
x, stats = dqgmres(A, b, M=opP, verbose=true)
show(stats)
r = b - A * x
@printf("Residual norm: %8.1e\n", norm(r))

# Right preconditioning
x, stats = dqgmres(A, b, N=opP, verbose=true)
show(stats)
r = b - A * x
@printf("Residual norm: %8.1e\n", norm(r))
