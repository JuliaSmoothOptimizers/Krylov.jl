using BenchmarkTools

using LinearAlgebra

using Krylov
using LinearOperators
using MatrixMarket

include("../test/get_div_grad.jl")
include("../test/test_utils.jl")
include("fetch_matrices.jl")

# we don't want to use matrixdepot(matrix, :read) to read in matrices because of
# https://github.com/JuliaMatrices/MatrixDepot.jl/issues/26

const SUITE = BenchmarkGroup()

SUITE["CG"] = BenchmarkGroup(["CG", "SPD"])

for N in [32, 64, 128]
  A = get_div_grad(N, N, N)
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  M = nonallocating_opEye(n)
  rtol = 1.0e-6
  SUITE["CG"]["DivGrad N=$N"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
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
  SUITE["CG"]["UFL-small"][matrix] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
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
  SUITE["CG"]["UFL-medium"][matrix] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
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
  SUITE["CG"]["UFL-large"][matrix] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
end
