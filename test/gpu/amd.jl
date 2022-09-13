using LinearAlgebra, SparseArrays, Test
using Krylov, AMDGPU

@testset "AMD -- AMDGPU.jl" begin

  @test AMDGPU.functional()
  AMDGPU.allowscalar(false)

  @testset "documentation" begin
    A_cpu = rand(ComplexF64, 20, 20)
    A_cpu = A_cpu + A_cpu'
    b_cpu = rand(ComplexF64, 20)
    A = A + A'
    A_gpu = ROCMatrix(A)
    b_gpu = ROCVector(b)
    x, stats = minres(A_gpu, b_gpu)
  end

  for FC in (Float32, Float64, ComplexF32, ComplexF64)
    S = ROCVector{FC}
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
    A = ROCMatrix{FC}(A)
    b = rand(FC, n)
    b = ROCVector{FC}(b)

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
