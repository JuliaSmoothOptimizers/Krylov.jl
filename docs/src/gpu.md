## GPU support

All solvers in Krylov.jl can be used with `CuArrays` and allow computations with Nvidia GPU. Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to GPU format (`CuMatrix` and `CuVector`).

```julia
using CUDA, Krylov

# CPU Arrays
A_cpu = rand(20, 20)
b_cpu = rand(20)

# GPU Arrays
A_gpu = CuMatrix(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a square and dense system on GPU
x, stats = bilq(A_gpu, b_gpu)
```

Sparse matrices have a specific storage on GPU (`CuSparseMatrixCSC` or `CuSparseMatrixCSR`):

```julia
using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

# CPU Arrays
A_cpu = sprand(200, 100, 0.3)
b_cpu = rand(200)

# GPU Arrays
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a rectangular and sparse system on GPU
x, stats = lsmr(A_gpu, b_gpu)
```

Optimized operator-vector products that exploit GPU features can be also used by means of linear operators.

Preconditioners, especially incomplete Cholesky or Incomplete LU factorizations that involve triangular solves,
can be applied directly on GPU thanks to efficient operators that take advantage of CUSPARSE routines.

### Example with a symmetric positive-definite system

```julia
using CUDA, Krylov, LinearOperators
using CUDA.CUSPARSE, SparseArrays

# LLᵀ ≈ A for CuSparseMatrixCSC matrices
P = ic02(A_gpu, 'O')

# Solve Py = x
function ldiv!(y, P, x)
  copyto!(y, x)                        # Variant for CuSparseMatrixCSR
  sv2!('T', 'U', 'N', 1.0, P, y, 'O')  # sv2!('N', 'L', 'N', 1.0, P, y, 'O')
  sv2!('N', 'U', 'N', 1.0, P, y, 'O')  # sv2!('T', 'L', 'N', 1.0, P, y, 'O')
  return y
end

# Operator that model P⁻¹
n = length(b_gpu)
T = eltype(b_gpu)
opM = LinearOperator(T, n, n, true, true, (y, x) -> ldiv!(y, P, x))

# Solve a symmetric positive definite system with an incomplete Cholesky preconditioner on GPU
(x, stats) = cg(A_gpu, b_gpu, M=opM)
```

### Example with a general square system

```julia
using CUDA, Krylov, LinearOperators
using CUDA.CUSPARSE, SparseArrays

# LU ≈ A for CuSparseMatrixCSC matrices
P = ilu02(A_gpu, 'O')

# Solve Py = x
function ldiv!(y, P, x)
  copyto!(y, x)                        # Variant for CuSparseMatrixCSR
  sv2!('N', 'L', 'N', 1.0, P, y, 'O')  # sv2!('N', 'L', 'U', 1.0, P, y, 'O')
  sv2!('N', 'U', 'U', 1.0, P, y, 'O')  # sv2!('N', 'U', 'N', 1.0, P, y, 'O')
  return y
end

# Operator that model P⁻¹
n = length(b_gpu)
T = eltype(b_gpu)
opM = LinearOperator(T, n, n, false, false, (y, x) -> ldiv!(y, P, x))

# Solve an unsymmetric system with an incomplete LU preconditioner on GPU
(x, stats) = bicgstab(A_gpu, b_gpu, M=opM)
```
