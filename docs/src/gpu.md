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

# CPU Arrays
A_cpu = rand(20, 20)
b_cpu = rand(20)

# GPU Arrays
A_gpu = CuMatrix(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a square and dense system on an Nivida GPU
x, stats = bilq(A_gpu, b_gpu)
```

Sparse matrices have a specific storage on Nvidia GPUs (`CuSparseMatrixCSC` or `CuSparseMatrixCSR`):

```julia
using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

# CPU Arrays
A_cpu = sprand(200, 100, 0.3)
b_cpu = rand(200)

# GPU Arrays
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a rectangular and sparse system on a Nvidia GPU
x, stats = lsmr(A_gpu, b_gpu)
```

Optimized operator-vector products that exploit GPU features can be also used by means of linear operators.

Preconditioners, especially incomplete Cholesky or Incomplete LU factorizations that involve triangular solves,
can be applied directly on GPU thanks to efficient operators that take advantage of CUSPARSE routines.

### Example with a symmetric positive-definite system

```julia
using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE

# Transfer the linear system from the CPU to the GPU
A_gpu = CuSparseMatrixCSC(A_cpu)  # A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)

# LLᴴ ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
P = ic02(A_gpu, 'O')

# Solve Py = x
function ldiv_ic0!(y, P, x)
  copyto!(y, x)                        # Variant for CuSparseMatrixCSR
  sv2!('T', 'U', 'N', 1.0, P, y, 'O')  # sv2!('N', 'L', 'N', 1.0, P, y, 'O')
  sv2!('N', 'U', 'N', 1.0, P, y, 'O')  # sv2!('T', 'L', 'N', 1.0, P, y, 'O')
  return y
end

# Operator that model P⁻¹
n = length(b_gpu)
T = eltype(b_gpu)
symmetric = hermitian = true
opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ic0!(y, P, x))

# Solve a symmetric positive definite system with an incomplete Cholesky preconditioner on GPU
x, stats = cg(A_gpu, b_gpu, M=opM)
```

!!! note
    You need to replace `'T'` by `'C'` in `ldiv_ic0!` if `A_gpu` is a complex matrix.

### Example with a general square system

```julia
using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

# Optional -- Compute a permutation vector p such that A[p,:] has no zero diagonal
p = zfd(A_cpu, 'O')
p .+= 1
A_cpu = A_cpu[p,:]
b_cpu = b_cpu[p]

# Transfer the linear system from the CPU to the GPU
A_gpu = CuSparseMatrixCSC(A_cpu)  # A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)

# LU ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
P = ilu02(A_gpu, 'O')

# Solve Py = x
function ldiv_ilu0!(y, P, x)
  copyto!(y, x)                        # Variant for CuSparseMatrixCSR
  sv2!('N', 'L', 'N', 1.0, P, y, 'O')  # sv2!('N', 'L', 'U', 1.0, P, y, 'O')
  sv2!('N', 'U', 'U', 1.0, P, y, 'O')  # sv2!('N', 'U', 'N', 1.0, P, y, 'O')
  return y
end

# Operator that model P⁻¹
n = length(b_gpu)
T = eltype(b_gpu)
symmetric = hermitian = false
opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ilu0!(y, P, x))

# Solve an unsymmetric system with an incomplete LU preconditioner on GPU
x, stats = bicgstab(A_gpu, b_gpu, M=opM)
```

## AMD GPUs

All solvers in Krylov.jl can be used with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and allow computations on AMD GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`ROCMatrix` and `ROCVector`).

```julia
using Krylov, AMDGPU

# CPU Arrays
A_cpu = rand(ComplexF64, 20, 20)
A_cpu = A_cpu + A_cpu'
b_cpu = rand(ComplexF64, 20)

A_gpu = ROCMatrix(A_cpu)
b_gpu = ROCVector(b_cpu)

# Solve a dense Hermitian system on an AMD GPU
x, stats = minres(A_gpu, b_gpu)
```

!!! info
    The library `rocSPARSE` is not interfaced yet in AMDGPU.jl and only dense linear systems are supported.

## Intel GPUs

All solvers in Krylov.jl, except [`MINRES-QLP`](@ref minres_qlp), can be used with [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) and allow computations on Intel GPUs.
Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to the related GPU format (`oneMatrix` and `oneVector`).

```julia
using Krylov, oneAPI

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
```

!!! warning
    The library `oneMKL` is not interfaced yet in oneAPI.jl and all BLAS routines (dot, norm, mul!, etc.) dispatch to generic fallbacks.

## Apple M1 GPUs

All solvers in Krylov.jl, except [`MINRES-QLP`](@ref minres_qlp), can be used with [Metal.jl](https://github.com/JuliaGPU/Metal.jl) and allow computations on Apple M1 GPUs.
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
