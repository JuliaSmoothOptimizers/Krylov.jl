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

Preconditioners, especially incomplete Cholesky or Incomplete LU factorizations with triangular solves,
can be performed directly on GPU with efficient operators that take advantage of CUSPARSE routines.

```julia
using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

# LLᵀ ≈ A
P = ic02(A_gpu, 'O')

# Solve Py = x
function ldiv!(y, P, x)
  y .= x
  sv2!('T', 'L', 1.0, P, y, 'O')
  sv2!('N', 'U', 1.0, P, y, 'O')
  return y
end

# Operator that model P⁻¹
y = similar(b_gpu); n = length(b_gpu); T = eltype(b_gpu)
opM = LinearOperator(T, n, n, true, true, x -> ldiv!(y, P, x))

# Solve a symmetric positive definite system with an incomplete Cholesky preconditioner on GPU
(x, stats) = cg(A_gpu, b_gpu, M=opM)
```

```julia
using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

# LU ≈ A
P = ilu02(A_gpu, 'O')

# Solve Py = x
function ldiv!(y, P, x)
  y .= x
  sv2!('N', 'L', 1.0, P, y, 'O')
  sv2!('N', 'U', 1.0, P, y, 'O', unit_diag=true)
  return y
end

# Operator that model P⁻¹
y = similar(b_gpu); n = length(b_gpu); T = eltype(b_gpu)
opN = LinearOperator(T, n, n, false, false, x -> ldiv!(y, P, x))

# Solve an unsymmetric system with an incomplete LU preconditioner on GPU
(x, stats) = bicgstab(A_gpu, b_gpu, M=opM)
```
