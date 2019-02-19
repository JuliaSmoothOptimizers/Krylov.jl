using BenchmarkTools
using MatrixMarket

using LinearAlgebra
using SparseArrays

using Krylov
using LinearOperators
using SuiteSparseMatrixCollection

include("../test/get_div_grad.jl")

# ufl_posdef = filter(p -> p.structure == "symmetric" && p.posDef == "yes" && p.type == "real" && p.rows ≤ 2_000, ssmc)
ufl_posdef = filter(p -> p.structure == "symmetric" && p.posDef == "yes" && p.type == "real" && p.rows ≤ 100, ssmc)

# fetch_ssmc(ufl_posdef, format="MM")

const SUITE = BenchmarkGroup()

SUITE["CG"] = BenchmarkGroup(["CG", "SPD"])

for N in [32, 64, 128]
  SUITE["CG"]["DivGrad N=$N"] = BenchmarkGroup()
  A = get_div_grad(N, N, N)
  n = size(A, 1)
  b = ones(n)
  op = PreallocatedLinearOperator(A)
  M = opEye()
  rtol = 1.0e-6
  SUITE["CG"]["DivGrad N=$N"]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
end

SUITE["CG"]["UFL"] = BenchmarkGroup()
for matrix in ufl_posdef
  name = matrix.name
  A = MatrixMarket.mmread(joinpath(matrix_path(matrix, format="MM"), "$(name).mtx"))
  if eltype(A) <: Integer
    A = convert(SparseMatrixCSC{Float64,Int}, A)
  end
  n = size(A, 1)
  b = ones(eltype(A), n)
  op = PreallocatedLinearOperator(A)
  M = opEye()
  rtol = 1.0e-6
  SUITE["CG"]["UFL"][name] = BenchmarkGroup()
  SUITE["CG"]["UFL"][name]["Krylov"] = @benchmarkable cg($op, $b, M=$M, atol=0.0, rtol=$rtol, itmax=$n)
end
