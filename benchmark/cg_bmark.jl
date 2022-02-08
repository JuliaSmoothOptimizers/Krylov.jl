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

ssmc = ssmc_db()
dataset = ssmc[(ssmc.numerical_symmetry .== 1) .& (ssmc.positive_definite.== true) .& (ssmc.real .== true) .& (ssmc.nrows .≤ 100), :]
paths = fetch_ssmc(dataset, format="MM")
names = dataset[!,:name]

const SUITE = BenchmarkGroup()

SUITE["CG"] = BenchmarkGroup(["CG", "SPD"])

for N in [32, 64, 128]
  SUITE["CG"]["DivGrad N=$N"] = BenchmarkGroup()
  A = get_div_grad(N, N, N)
  n, m = size(A)
  b = ones(n)
  rtol = 1.0e-6
  SUITE["CG"]["DivGrad N=$N"]["Krylov"] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
  SUITE["CG"]["DivGrad N=$N"]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["DivGrad N=$N"]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, abstol=0.0, reltol=$rtol, maxiter=$n)
  SUITE["CG"]["DivGrad N=$N"]["KrylovKit"] = @benchmarkable ($KrylovKit.linsolve)($A, $b, atol=0.0, rtol=$rtol, isposdef=true, krylovdim=$n, maxiter=1)
end

SUITE["CG"]["UFL"] = BenchmarkGroup()
nb_pbs = length(paths)
for i = 1 : nb_pbs
  name = dataset[!,:name][i]
  path = paths[i]
  A = Float64.(MatrixMarket.mmread(path * "/$name.mtx"))
  n, m = size(A)
  b = ones(n)
  rtol = 1.0e-6
  SUITE["CG"]["UFL"][name] = BenchmarkGroup()
  SUITE["CG"]["UFL"][name]["Krylov"] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
  SUITE["CG"]["UFL"][name]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["UFL"][name]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, abstol=0.0, reltol=$rtol, maxiter=$n)
  SUITE["CG"]["UFL"][name]["KrylovKit"] = @benchmarkable ($KrylovKit.linsolve)($A, $b, atol=0.0, rtol=$rtol, isposdef=true, krylovdim=$n, maxiter=1)
end
