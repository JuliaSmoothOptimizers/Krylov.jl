using PkgBenchmark
using BenchmarkTools
using MatrixMarket

using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE

using Krylov
using LinearOperators
using SuiteSparseMatrixCollection

ufl_bicgstab = filter(p -> p.structure == "unsymmetric" && p.type == "real" && (10000 ≤ p.rows ≤ 20000), ssmc)
ufl_cg = filter(p -> p.structure == "symmetric" && p.posDef == "yes" && p.type == "real" && (10000 ≤ p.rows ≤ 20000), ssmc)

# fetch_ssmc(ufl_cg, format="MM")
# fetch_ssmc(ufl_bicgstab, format="MM")

const SUITE = BenchmarkGroup()

SUITE["GPU"] = BenchmarkGroup(["CG", "BICGSTAB"])

SUITE["GPU"]["CG"] = BenchmarkGroup()
for matrix in ufl_cg
  name = matrix.name
  A = MatrixMarket.mmread(joinpath(matrix_path(matrix, format="MM"), "$(name).mtx"))
  A = CuSparseMatrixCSC{Float64}(A)
  n = size(A, 1)
  b = ones(n)
  b = CuVector(b)
  rtol = 1.0e-8
  SUITE["GPU"]["CG"][name] = @benchmarkable CUDA.@sync cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end

SUITE["GPU"]["BICGSTAB"] = BenchmarkGroup()
for matrix in ufl_bicgstab
  name = matrix.name
  A = MatrixMarket.mmread(joinpath(matrix_path(matrix, format="MM"), "$(name).mtx"))
  A = CuSparseMatrixCSC{Float64}(A)
  n = size(A, 1)
  b = ones(n)
  b = CuVector(b)
  rtol = 1.0e-8
  SUITE["GPU"]["BICGSTAB"][name] = @benchmarkable CUDA.@sync bicgstab($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end
