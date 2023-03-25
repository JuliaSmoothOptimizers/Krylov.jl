using LinearAlgebra                # Linear algebra library of Julia
using SparseArrays                 # Sparse library of Julia
using Krylov                       # Krylov methods and processes
using LinearOperators              # Linear operators
using MatrixMarket                 # Reader of matrices stored in the Matrix Market format
using SuiteSparseMatrixCollection  # Interface to the SuiteSparse Matrix Collection
using CUDA                         # Interface to NVIDIA GPUs
using CUDA.CUSPARSE                # NVIDIA CUSPARSE library

if CUDA.functional()
  ssmc = ssmc_db()
  matrices = ssmc_matrices(ssmc, "Bai", "mhd1280b")
  paths = fetch_ssmc(matrices, format="MM")
  path_A = joinpath(paths[1], "mhd1280b.mtx")
  A_cpu = MatrixMarket.mmread(path_A)
  m, n = size(A_cpu)
  b_cpu = ones(ComplexF64, m)

  # Transfer the linear system from the CPU to the GPU
  A_gpu = CuSparseMatrixCSR(A_cpu)
  b_gpu = CuVector(b_cpu)

  # Incomplete Cholesky decomposition LLᴴ ≈ A with zero fill-in
  P = ic02(A_gpu, 'O')

  # Additional vector required for solving triangular systems
  z = similar(CuVector{ComplexF64}, n)

  # Solve Py = x
  function ldiv_ic0!(P, x, y, z)
    ldiv!(z, LowerTriangular(P), x)   # Forward substitution with L
    ldiv!(y, LowerTriangular(P)', z)  # Backward substitution with Lᴴ
    return y
  end

  # Linear operator that approximates the preconditioner P⁻¹ in floating-point arithmetic
  T = ComplexF64
  symmetric = false
  hermitian = true
  P⁻¹ = LinearOperator(T, m, n, symmetric, hermitian, (y, x) -> ldiv_ic0!(P, x, y, z))

  # Solve a Hermitian positive definite system with an incomplete Cholesky factorization preconditioner
  x, stats = cg(A_gpu, b_gpu, M=P⁻¹)
end
