using Krylov, LinearOperators, IncompleteLU
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
F = ilu(A, τ = 0.1)

@printf("nnz(ILU) / nnz(A): %7.1e\n", nnz(F) / nnz(A))

# Solve Ax = b with CGS and an ILU preconditioner
yM = zeros(N)
yN = zeros(N)
opM = LinearOperator(Float64, N, N, false, false, y -> (yM .= y ; IncompleteLU.forward_substitution_without_diag!(F.L, yM)))
opN = LinearOperator(Float64, N, N, false, false, y -> (yN .= y ; IncompleteLU.transposed_backward_substitution!(F.U, yN)))

x, stats = cgs(A, b, M=opM, N=opN, verbose=true)
show(stats)
r = b - A * x
@printf("Residual norm: %8.1e\n", norm(r))
