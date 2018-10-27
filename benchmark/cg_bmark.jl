using BenchmarkTools

using LinearAlgebra

import IterativeSolvers
# import KrylovMethods
using Krylov
using LinearOperators
using MatrixMarket

include("../test/get_div_grad.jl")
include("../test/test_utils.jl")
include("fetch_matrices.jl")

const SUITE = BenchmarkGroup()

SUITE["CG"] = BenchmarkGroup(["CG", "SPD"])

for N in [32, 64, 128]
  SUITE["CG"]["DivGrad N=$N"] = BenchmarkGroup()
  A = get_div_grad(N, N, N)
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  M = nonallocating_opEye(n)
  rtol = 1.0e-6
  SUITE["CG"]["DivGrad N=$N"]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  # SUITE["CG"]["DivGrad N=$N"]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["DivGrad N=$N"]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, tol=$rtol, maxiter=$n)
end

SUITE["CG"]["UFL-small"] = BenchmarkGroup()
for matrix in spd_small
  name = basename(matrix)
  A = MatrixMarket.mmread(joinpath(matrix_path, "..", "data", "uf", matrix, "$(name).mtx"))
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  M = nonallocating_opEye(n)
  rtol = 1.0e-6
  SUITE["CG"]["UFL-small"][matrix] = BenchmarkGroup()
  SUITE["CG"]["UFL-small"][matrix]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  # SUITE["CG"]["UFL-small"][matrix]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["UFL-small"][matrix]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, tol=$rtol, maxiter=$n)
end

SUITE["CG"]["UFL-medium"] = BenchmarkGroup()
for matrix in spd_med
  name = basename(matrix)
  A = MatrixMarket.mmread(joinpath(matrix_path, "..", "data", "uf", matrix, "$(name).mtx"))
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  M = nonallocating_opEye(n)
  rtol = 1.0e-6
  SUITE["CG"]["UFL-medium"][matrix] = BenchmarkGroup()
  SUITE["CG"]["UFL-medium"][matrix]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  # SUITE["CG"]["UFL-medium"][matrix]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["UFL-medium"][matrix]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, tol=$rtol, maxiter=$n)
end

SUITE["CG"]["UFL-large"] = BenchmarkGroup()
for matrix in spd_large
  name = basename(matrix)
  A = MatrixMarket.mmread(joinpath(matrix_path, "..", "data", "uf", matrix, "$(name).mtx"))
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  M = nonallocating_opEye(n)
  rtol = 1.0e-6
  SUITE["CG"]["UFL-large"][matrix] = BenchmarkGroup()
  SUITE["CG"]["UFL-large"][matrix]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  # SUITE["CG"]["UFL-large"][matrix]["KrylovMethods"] = @benchmarkable ($KrylovMethods.cg)($A, $b, tol=$rtol, maxIter=$n)
  SUITE["CG"]["UFL-large"][matrix]["IterativeSolvers"] = @benchmarkable ($IterativeSolvers.cg)($A, $b, tol=$rtol, maxiter=$n)
end
