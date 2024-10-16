using oneAPI

include("gpu.jl")

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

  for FC ∈ (Float32, ) # ComplexF32)
    S = oneVector{FC}
    M = oneMatrix{FC}
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
      Krylov.@kcopy!(n, y, x)
    end

    @testset "kswap -- $FC" begin
      Krylov.@kswap(x, y)
    end

    @testset "kref! -- $FC" begin
      Krylov.@kref!(n, x, y, c, s)
    end

    @testset "conversion -- $FC" begin
      test_conversion(S, M)
    end

    ε = eps(T)
    atol = √ε
    rtol = √ε

    @testset "GMRES -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = gmres(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "block-GMRES -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      B = hcat(b, -b)
      A = M(A)
      B = M(B)
      X, stats = block_gmres(A, B)
      @test norm(B - A * X) ≤ atol + rtol * norm(B)
    end

    @testset "CG -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = cg(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    # @testset "MINRES-QLP -- $FC" begin
    #   A, b = symmetric_indefinite(FC=FC)
    #   A = M(A)
    #   b = S(b)
    #   x, stats = minres_qlp(A, b)
    #   @test norm(b - A * x) ≤ atol + rtol * norm(b)
    # end

    # @testset "processes -- $FC" begin
    #   test_processes(S, M)
    # end

    @testset "solver -- $FC" begin
      test_solver(S, M)
    end
  end
end
