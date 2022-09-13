using LinearAlgebra, SparseArrays, Test
using Krylov, oneAPI

include("../test_utils.jl")

import Krylov.kdot
# https://github.com/JuliaGPU/GPUArrays.jl/pull/427
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
    atol = √ε
    rtol = √ε

    @testset "GMRES -- $FC" begin
      A, b = nonsymmetric_indefinite(FC=FC)
      A = oneMatrix{FC}(A)
      b = oneVector{FC}(b)
      x, stats = gmres(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "CG -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      A = oneMatrix{FC}(A)
      b = oneVector{FC}(b)
      x, stats = cg(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end
  end
end
