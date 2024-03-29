using BenchmarkTools
import IterativeSolvers
import KrylovKit
import KrylovMethods

using LinearAlgebra
using Printf
using SparseArrays

using Krylov
using MatrixMarket
using SuiteSparseMatrixCollection

krylov_problem = joinpath(dirname(pathof(Krylov)), "..", "test", "get_div_grad.jl")
include(krylov_problem)

ssmc = ssmc_db(verbose=false)
dataset = ssmc[(ssmc.numerical_symmetry .== 1) .& (ssmc.positive_definite.== true) .& (ssmc.real .== true) .& (3000 .≤ ssmc.nrows .≤ 5000), :]
paths = fetch_ssmc(dataset, format="MM")
names = dataset[!,:name]

const SUITE = BenchmarkGroup()

SUITE["Krylov"] = BenchmarkGroup()
SUITE["KrylovMethods"] = BenchmarkGroup()
SUITE["IterativeSolvers"] = BenchmarkGroup()
SUITE["KrylovKit"] = BenchmarkGroup()

for N in [32, 64, 128]
  A = get_div_grad(N, N, N)
  n, m = size(A)
  b = ones(n)
  rtol = 1.0e-6
  SUITE["Krylov"]["DivGrad N=$N"] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
  SUITE["KrylovMethods"]["DivGrad N=$N"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["IterativeSolvers"]["DivGrad N=$N"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, abstol=0.0, reltol=$rtol, maxiter=$n)
  SUITE["KrylovKit"]["DivGrad N=$N"] = @benchmarkable ($KrylovKit.linsolve)($A, $b, atol=0.0, rtol=$rtol, isposdef=true, krylovdim=$n, maxiter=1)
end

nb_pbs = length(paths)
for i = 1 : nb_pbs
  name = dataset[!,:name][i]
  path = paths[i]
  A = MatrixMarket.mmread(path * "/$name.mtx")
  n, m = size(A)
  if eltype(A) == Float64
    b = ones(n)
    rtol = 1.0e-6
    SUITE["Krylov"][name] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
    SUITE["KrylovMethods"][name] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
    SUITE["IterativeSolvers"][name] = @benchmarkable ($IterativeSolvers.cg)($A, $b, abstol=0.0, reltol=$rtol, maxiter=$n)
    SUITE["KrylovKit"][name] = @benchmarkable ($KrylovKit.linsolve)($A, $b, atol=0.0, rtol=$rtol, isposdef=true, krylovdim=$n, maxiter=1)
  end
end
