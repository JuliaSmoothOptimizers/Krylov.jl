using BenchmarkTools

using LinearAlgebra
using Logging

using Krylov
using LinearOperators
using MatrixMarket

cg_logger = Logging.NullLogger()  # no output

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
  SUITE["CG"]["DivGrad N=$N"] = @benchmarkable with_logger(cg_logger) do
    cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  end
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
  SUITE["CG"]["UFL-small"][matrix] = @benchmarkable with_logger(cg_logger) do
    cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  end
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
  SUITE["CG"]["UFL-medium"][matrix] = @benchmarkable with_logger(cg_logger) do
    cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  end
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
  SUITE["CG"]["UFL-large"][matrix] = @benchmarkable with_logger(cg_logger) do
    cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
  end
end
