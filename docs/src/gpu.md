# [GPU support](@id gpu)

Krylov methods are well suited for GPU computations because they only require matrix-vector products ($u \leftarrow Av$, $u \leftarrow A^{H}w$) and vector operations ($\|v\|$, $u^H v$, $v \leftarrow \alpha u + \beta v$), which are highly parallelizable.

The implementations in Krylov.jl are generic so as to take advantage of the multiple dispatch and broadcast features of Julia.
Those allow the implementations to be specialized automatically by the compiler for both CPU and GPU.
Thus, Krylov.jl works with GPU backends that build on [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl), such as [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) or [Metal.jl](https://github.com/JuliaGPU/Metal.jl).

## Nvidia GPUs

All solvers in Krylov.jl can be used with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and allow computations on Nvidia GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`CuMatrix` and `CuVector`).

```julia
using CUDA, Krylov

if CUDA.functional()
  # CPU Arrays
  A_cpu = rand(20, 20)
  b_cpu = rand(20)

  # GPU Arrays
  A_gpu = CuMatrix(A_cpu)
  b_gpu = CuVector(b_cpu)

  # Solve a square and dense system on an Nivida GPU
  x, stats = bilq(A_gpu, b_gpu)
end
```

Sparse matrices have a specific storage on Nvidia GPUs (`CuSparseMatrixCSC`, `CuSparseMatrixCSR` or `CuSparseMatrixCOO`):

```julia
using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

if CUDA.functional()
  # CPU Arrays
  A_cpu = sprand(200, 100, 0.3)
  b_cpu = rand(200)

  # GPU Arrays
  A_csc_gpu = CuSparseMatrixCSC(A_cpu)
  A_csr_gpu = CuSparseMatrixCSR(A_cpu)
  A_coo_gpu = CuSparseMatrixCOO(A_cpu)
  b_gpu = CuVector(b_cpu)

  # Solve a rectangular and sparse system on an Nvidia GPU
  x_csc, stats_csc = lslq(A_csc_gpu, b_gpu)
  x_csr, stats_csr = lsqr(A_csr_gpu, b_gpu)
  x_coo, stats_coo = lsmr(A_coo_gpu, b_gpu)
end
```

If you use a Krylov method that only requires `A * v` products (see [here](@ref factorization-free)), the most efficient format is `CuSparseMatrixCSR`.
Optimized operator-vector products that exploit GPU features can be also used by means of linear operators.

Preconditioners, especially incomplete Cholesky or Incomplete LU factorizations that involve triangular solves,
can be applied directly on GPU thanks to efficient operators that take advantage of CUSPARSE routines.

### Example with a symmetric positive-definite system

```julia
using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE

if CUDA.functional()
  # Transfer the linear system from the CPU to the GPU
  A_gpu = CuSparseMatrixCSR(A_cpu)  # A_gpu = CuSparseMatrixCSC(A_cpu)
  b_gpu = CuVector(b_cpu)

  # IC(0) decomposition LLᴴ ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
  P = ic02(A_gpu)

  # Additional vector required for solving triangular systems
  n = length(b_gpu)
  T = eltype(b_gpu)
  z = CUDA.zeros(T, n)

  # Solve Py = x
  function ldiv_ic0!(P::CuSparseMatrixCSR, x, y, z)
    ldiv!(z, LowerTriangular(P), x)   # Forward substitution with L
    ldiv!(y, LowerTriangular(P)', z)  # Backward substitution with Lᴴ
    return y
  end

  function ldiv_ic0!(P::CuSparseMatrixCSC, x, y, z)
    ldiv!(z, UpperTriangular(P)', x)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)   # Backward substitution with Lᴴ
    return y
  end

  # Operator that model P⁻¹
  symmetric = hermitian = true
  opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ic0!(P, x, y, z))

  # Solve an Hermitian positive definite system with an IC(0) preconditioner on GPU
  x, stats = cg(A_gpu, b_gpu, M=opM)
end
```

### Example with a general square system

```julia
using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

if CUDA.functional()
  # Optional -- Compute a permutation vector p such that A[:,p] has no zero diagonal
  p = zfd(A_cpu)
  p .+= 1
  A_cpu = A_cpu[:,p]

  # Transfer the linear system from the CPU to the GPU
  A_gpu = CuSparseMatrixCSR(A_cpu)  # A_gpu = CuSparseMatrixCSC(A_cpu)
  b_gpu = CuVector(b_cpu)

  # ILU(0) decomposition LU ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
  P = ilu02(A_gpu)

  # Additional vector required for solving triangular systems
  n = length(b_gpu)
  T = eltype(b_gpu)
  z = CUDA.zeros(T, n)

  # Solve Py = x
  function ldiv_ilu0!(P::CuSparseMatrixCSR, x, y, z)
    ldiv!(z, UnitLowerTriangular(P), x)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)      # Backward substitution with U
    return y
  end

  function ldiv_ilu0!(P::CuSparseMatrixCSC, x, y, z)
    ldiv!(z, LowerTriangular(P), x)      # Forward substitution with L
    ldiv!(y, UnitUpperTriangular(P), z)  # Backward substitution with U
    return y
  end

  # Operator that model P⁻¹
  symmetric = hermitian = false
  opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P, x, y, z))

  # Solve a non-Hermitian system with an ILU(0) preconditioner on GPU
  x̄, stats = bicgstab(A_gpu, b_gpu, M=opM)

  # Recover the solution of Ax = b with the solution of A[:,p]x̄ = b
  invp = invperm(p)
  x = x̄[invp]
end
```

## AMD GPUs

All solvers in Krylov.jl can be used with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and allow computations on AMD GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`ROCMatrix` and `ROCVector`).

```julia
using Krylov, AMDGPU

if AMDGPU.functional()
  # CPU Arrays
  A_cpu = rand(ComplexF64, 20, 20)
  A_cpu = A_cpu + A_cpu'
  b_cpu = rand(ComplexF64, 20)

  A_gpu = ROCMatrix(A_cpu)
  b_gpu = ROCVector(b_cpu)

  # Solve a dense Hermitian system on an AMD GPU
  x, stats = minres(A_gpu, b_gpu)
end
```

Sparse matrices have a specific storage on AMD GPUs (`ROCSparseMatrixCSC`, `ROCSparseMatrixCSR` or `ROCSparseMatrixCOO`):

```julia
using AMDGPU, Krylov
using AMDGPU.rocSPARSE, SparseArrays

if AMDGPU.functional()
  # CPU Arrays
  A_cpu = sprand(100, 200, 0.3)
  b_cpu = rand(100)

  # GPU Arrays
  A_csc_gpu = ROCSparseMatrixCSC(A_cpu)
  A_csr_gpu = ROCSparseMatrixCSR(A_cpu)
  A_coo_gpu = ROCSparseMatrixCOO(A_cpu)
  b_gpu = CuVector(b_cpu)

  # Solve a rectangular and sparse system on an AMD GPU
  x_csc, y_csc, stats_csc = lnlq(A_csc_gpu, b_gpu)
  x_csr, y_csr, stats_csr = craig(A_csr_gpu, b_gpu)
  x_coo, y_coo, stats_coo = craigmr(A_coo_gpu, b_gpu)
end
```

## Intel GPUs

All solvers in Krylov.jl can be used with [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) and allow computations on Intel GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`oneMatrix` and `oneVector`).

```julia
using Krylov, oneAPI

if oneAPI.functional()
  T = Float32  # oneAPI.jl also works with ComplexF32
  m = 20
  n = 10

  # CPU Arrays
  A_cpu = rand(T, m, n)
  b_cpu = rand(T, m)

  # GPU Arrays
  A_gpu = oneMatrix(A_cpu)
  b_gpu = oneVector(b_cpu)

  # Solve a dense least-squares problem on an Intel GPU
  x, stats = lsqr(A_gpu, b_gpu)
end
```

!!! note
    The library `oneMKL` is interfaced in oneAPI.jl and accelerates linear algebra operations on Intel GPUs. Only dense linear systems are supported for the time being because sparse linear algebra routines are not interfaced yet.

## Apple M1 GPUs

All solvers in Krylov.jl can be used with [Metal.jl](https://github.com/JuliaGPU/Metal.jl) and allow computations on Apple M1 GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`MtlMatrix` and `MtlVector`).

```julia
using Krylov, Metal

T = Float32  # Metal.jl also works with ComplexF32
n = 10
m = 20

# CPU Arrays
A_cpu = rand(T, n, m)
b_cpu = rand(T, n)

# GPU Arrays
A_gpu = MtlMatrix(A_cpu)
b_gpu = MtlVector(b_cpu)

# Solve a dense least-norm problem on an Apple M1 GPU
x, stats = craig(A_gpu, b_gpu)
```

!!! warning
    Metal.jl is under heavy development and is considered experimental for now.
