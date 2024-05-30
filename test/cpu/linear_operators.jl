using LinearAlgebra, SparseArrays, Test
using Krylov, LinearOperators

@testset "LinearOperators" begin
  n = 50
  p = 5

  for T in (Float64, ComplexF64)
    A = rand(T, n, n)
    B = rand(T, n, p)
    
    opA = LinearOperator(A)
    x, stats = block_gmres(opA, B)
    @test stats.solved
  end
end
