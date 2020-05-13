## GPU support

All solvers in Krylov.jl can be used with `CuArrays` and allow computations with Nvidia GPU. Problems stored in CPU format (`Matrix` and `Vector`) must first be converted to GPU format (`CuMatrix` and `CuVector`).

```julia
using CuArrays, Krylov

# CPU Arrays
A_cpu = rand(20, 20)
b_cpu = rand(20)

# GPU Arrays
A_gpu = CuMatrix(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a square and dense system on GPU
x, stats = dqgmres(A_gpu, b_gpu)
```

Sparse matrices have a specific storage on GPU (`CuSparseMatrixCSC` or `CuSparseMatrixCSR`):

```julia
using CuArrays, Krylov
using CuArrays.CUSPARSE

# CPU Arrays
A_cpu = sprand(200, 100, 0.3)
b_cpu = rand(200)

# GPU Arrays
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuVector(b_cpu)

# Solve a rectangular and sparse system on GPU
x, stats = lsmr(A_gpu, b_gpu)
```

!!! note
    Krylov.jl requires at least CuArrays.jl v2.2.0.
