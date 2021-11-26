using BenchmarkTools
using LinearAlgebra, SparseArrays
using MatrixMarket, SuiteSparseMatrixCollection

include("../test/get_div_grad.jl")

ssmc = ssmc_db()
ufl_posdef = ssmc[(ssmc.numerical_symmetry .== 1) .& (ssmc.positive_definite .== true) .& (ssmc.real .== true) .& (ssmc.binary .== false) .& (ssmc.nrows .≤ 500), :]
paths = fetch_ssmc(ufl_posdef, format="MM")

const SUITE = BenchmarkGroup()

SUITE["DivGrad"] = BenchmarkGroup()
for N in [32, 64, 128]
  A = get_div_grad(N, N, N)
  n, m = size(A)
  b = ones(n)
  rtol = 1.0e-8
  SUITE["DivGrad"]["DivGrad N=$N"] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end

SUITE["UFL"] = BenchmarkGroup()
for path in paths
  name = split(path, '/')[end]
  A = MatrixMarket.mmread(path * "/$name.mtx")
  n, m = size(A)
  b = ones(n)
  rtol = 1.0e-8
  SUITE["UFL"][name] = @benchmarkable cg($A, $b, atol=0.0, rtol=$rtol, itmax=$n)
end
