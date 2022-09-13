using LinearAlgebra, SparseArrays, Test
using Krylov, oneAPI

# https://github.com/JuliaGPU/GPUArrays.jl/pull/427
import Krylov.kdot
function kdot(n :: Integer, x :: oneVector{T}, dx :: Integer, y :: oneVector{T}, dy :: Integer) where T <: Krylov.FloatOrComplex
  return mapreduce(dot, +, x, y)
end

@testset "Intel -- oneAPI.jl" begin

  @test oneAPI.functional()
  oneAPI.allowscalar(false)

  @testset "documentation" begin
    T = Float32
    m = 20
    n = 10
    A_cpu = rand(T, m, n)
    b_cpu = rand(T, m)
    A_gpu = oneMatrix(A_cpu)
    b_gpu = oneVector(b_cpu)
    x, stats = lsqr(A_gpu, b_gpu)
  end

  for FC ∈ (Float32, ComplexF32)
    S = oneVector{FC}
    T = real(FC)
    n = 10
    x = rand(FC, n)
    x = S(x)
    y = rand(FC, n)
    y = S(y)
    a = rand(FC)
    b = rand(FC)
    s = rand(FC)
    a2 = rand(T)
    b2 = rand(T)
    c = rand(T)

    @testset "kdot -- $FC" begin
      Krylov.@kdot(n, x, y)
    end

    @testset "kdotr -- $FC" begin
      Krylov.@kdotr(n, x, y)
    end

    @testset "knrm2 -- $FC" begin
      Krylov.@knrm2(n, x)
    end

    @testset "kaxpy! -- $FC" begin
      Krylov.@kaxpy!(n, a, x, y)
      Krylov.@kaxpy!(n, a2, x, y)
    end

    @testset "kaxpby! -- $FC" begin
      Krylov.@kaxpby!(n, a, x, b, y)
      Krylov.@kaxpby!(n, a2, x, b, y)
      Krylov.@kaxpby!(n, a, x, b2, y)
      Krylov.@kaxpby!(n, a2, x, b2, y)
    end

    @testset "kcopy! -- $FC" begin
      Krylov.@kcopy!(n, x, y)
    end

    @testset "kswap -- $FC" begin
      Krylov.@kswap(x, y)
    end

    # @testset "kref! -- $FC" begin
    #   Krylov.@kref!(n, x, y, c, s)
    # end

    ε = eps(T)
    A = rand(FC, n, n)
    A = oneMatrix{FC}(A)
    b = rand(FC, n)
    b = oneVector{FC}(b)

    @testset "GMRES -- $FC" begin
      x, stats = gmres(A, b)
      @test norm(b - A * x) ≤ √ε
    end

    @testset "CG -- $FC" begin
      C = A * A'
      x, stats = cg(C, b)
      @test stats.solved
    end
  end
end
