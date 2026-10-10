using PkgBenchmark
using BenchmarkTools
using MatrixMarket

using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.cuSPARSE

using Krylov
using LinearOperators
using SuiteSparseMatrixCollection

ssmc = ssmc_db(verbose=false)
ufl_bicgstab = ssmc[(ssmc.nrows .== ssmc.ncols) .& (ssmc.numerical_symmetry .< 1) .& (ssmc.real .== true) .& (ssmc.binary .== false) .& (10000 .≤ ssmc.nrows .≤ 20000), :]
ufl_cg = ssmc[(ssmc.numerical_symmetry .== 1) .& (ssmc.positive_definite .== true) .& (ssmc.real .== true) .& (ssmc.binary .== false) .& (10000 .≤ ssmc.nrows .≤ 20000), :]

paths_bicgstab = fetch_ssmc(ufl_bicgstab, format="MM")
paths_cg = fetch_ssmc(ufl_cg, format="MM")

const SUITE = BenchmarkGroup()

SUITE["GPU"] = BenchmarkGroup(["CG", "BICGSTAB"])

SUITE["GPU"]["CG"] = BenchmarkGroup()
for (name, path) in zip(ufl_cg.name, paths_cg)
  A = MatrixMarket.mmread(joinpath(path, "$(name).mtx"))
  A = CuSparseMatrixCSC{Float64}(A)
  n = size(A, 1)
  b = ones(n)
  b = CuVector(b)
  rtol = 1.0e-8
  SUITE["GPU"]["CG"][name] = @benchmarkable CUDA.@sync cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end

SUITE["GPU"]["BICGSTAB"] = BenchmarkGroup()
for (name, path) in zip(ufl_bicgstab.name, paths_bicgstab)
  A = MatrixMarket.mmread(joinpath(path, "$(name).mtx"))
  A = CuSparseMatrixCSC{Float64}(A)
  n = size(A, 1)
  b = ones(n)
  b = CuVector(b)
  rtol = 1.0e-8
  SUITE["GPU"]["BICGSTAB"][name] = @benchmarkable CUDA.@sync bicgstab($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end
