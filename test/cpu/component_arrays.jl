using LinearAlgebra, SparseArrays, Test
using Krylov, ComponentArrays

@testset "ComponentArrays" begin
  n = 5

  for T in (Float32, Float64)
    A = rand(T, n, n)

    b = ComponentArrays{T}(a=[1, 2, 3], b=[4, 5])
    @test Krylov.ktypeof(b) == Vector{T}
    x, stats = gmres(A, b)
    @test stats.solved
    
  end
end
