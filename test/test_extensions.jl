using LinearAlgebra, SparseArrays, Test
using Krylov, StaticArrays

@testset "StaticArrays" begin
  n = 5

  for T in (Float32, Float64)
    A = rand(T, n, n)

    b = SVector{n}(rand(T, n))
    @test Krylov.ktypeof(b) == Vector{T}
    x, stats = gmres(A, b)
    @test stats.solved

    b = MVector{n}(rand(T, n))
    @test Krylov.ktypeof(b) == Vector{T}
    x, stats = gmres(A, b)
    @test stats.solved

    b = SizedVector{n}(rand(T, n))
    @test Krylov.ktypeof(b) == Vector{T}
    x, stats = gmres(A, b)
    @test stats.solved
  end
end
