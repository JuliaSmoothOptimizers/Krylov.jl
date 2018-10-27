using BenchmarkTools

using LinearAlgebra

using Krylov
using LinearOperators

include("../test/get_div_grad.jl")
include("../test/test_utils.jl")

const SUITE = BenchmarkGroup()

SUITE["CG"] = BenchmarkGroup(["CG", "SPD"])

for N in [32, 64, 128]
  A = get_div_grad(N, N, N)
  n = size(A, 1)
  b = ones(n)
  op = preallocated_LinearOperator(A)
  SUITE["CG"]["DivGrad N=$N"] = @benchmarkable cg($op, $b)
end
