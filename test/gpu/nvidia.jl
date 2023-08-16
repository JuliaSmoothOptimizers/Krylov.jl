using LinearOperators, CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

include("gpu.jl")

@testset "Nvidia -- CUDA.jl" begin

  @test CUDA.functional()
  CUDA.allowscalar(false)

  @testset "documentation" begin
    A_cpu = rand(20, 20)
    b_cpu = rand(20)
    A_gpu = CuMatrix(A_cpu)
    b_gpu = CuVector(b_cpu)
    x, stats = bilq(A_gpu, b_gpu)

    A_cpu = sprand(200, 100, 0.3)
    b_cpu = rand(200)
    A_csc_gpu = CuSparseMatrixCSC(A_cpu)
    A_csr_gpu = CuSparseMatrixCSR(A_cpu)
    A_coo_gpu = CuSparseMatrixCOO(A_cpu)
    b_gpu = CuVector(b_cpu)
    b_gpu = CuVector(b_cpu)
    x_csc, stats_csc = lslq(A_csc_gpu, b_gpu)
    Aᴴr_csc = A_csc_gpu' * (b_gpu - A_csc_gpu * x_csc)
    println("cas 1")
    println(norm(Aᴴr_csc))
    x_csr, stats_csr = lsqr(A_csr_gpu, b_gpu)
    Aᴴr_csr = A_csr_gpu' * (b_gpu - A_csr_gpu * x_csr)
    println("cas 2")
    println(norm(Aᴴr_csr))
    x_coo, stats_coo = lsmr(A_coo_gpu, b_gpu)
    Aᴴr_coo = A_coo_gpu' * (b_gpu - A_coo_gpu * x_coo)
    println("cas 3")
    println(norm(Aᴴr_coo))

    @testset "ic0" begin
      A_cpu, b_cpu = sparse_laplacian()
      @test mapreduce(Aᵢᵢ -> Aᵢᵢ != 0, &, diag(A_cpu)) == true

      b_gpu = CuVector(b_cpu)
      n = length(b_gpu)
      T = eltype(b_gpu)
      z = CUDA.zeros(T, n)
      symmetric = hermitian = true

      A_gpu = CuSparseMatrixCSC(A_cpu)
      P = ic02(A_gpu)
      function ldiv_ic0!(P::CuSparseMatrixCSC, x, y, z)
        ldiv!(z, UpperTriangular(P)', x)
        ldiv!(y, UpperTriangular(P), z)
        return y
      end
      opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ic0!(P, x, y, z))
      x, stats = cg(A_gpu, b_gpu, M=opM)
      @test norm(b_gpu - A_gpu * x) ≤ 1e-6
      @test stats.niter ≤ 19

      A_gpu = CuSparseMatrixCSR(A_gpu)
      P = ic02(A_gpu)
      function ldiv_ic0!(P::CuSparseMatrixCSR, x, y, z)
        ldiv!(z, LowerTriangular(P), x)
        ldiv!(y, LowerTriangular(P)', z)
        return y
      end
      opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ic0!(P, x, y, z))
      x, stats = cg(A_gpu, b_gpu, M=opM)
      @test norm(b_gpu - A_gpu * x) ≤ 1e-6
      @test stats.niter ≤ 19
    end

    @testset "ilu0" begin
      A_cpu = Float64[1  0 0  4;
                      0  0 7  8;
                      9  0 0 12;
                      0 14 0 16]
      A_cpu = sparse(A_cpu)
      b_cpu = ones(4)
      @test mapreduce(Aᵢᵢ -> Aᵢᵢ != 0, &, diag(A_cpu)) == false

      p = zfd(A_cpu)
      p .+= 1
      invp = invperm(p)
      @test reduce(&, invp .== p) == false

      b_gpu = CuVector(b_cpu)
      n = length(b_gpu)
      T = eltype(b_gpu)
      z = CUDA.zeros(T, n)
      symmetric = hermitian = false

      A_gpu = CuSparseMatrixCSC(A_cpu[:,p])
      P = ilu02(A_gpu)
      function ldiv_ilu0!(P::CuSparseMatrixCSC, x, y, z)
        ldiv!(z, LowerTriangular(P), x)
        ldiv!(y, UnitUpperTriangular(P), z)
        return y
      end
      opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P, x, y, z))
      x̄, stats = gmres(A_gpu, b_gpu, M=opM)
      x = Vector(x̄)[invp]
      @test norm(b_gpu - A_gpu * x̄) ≤ 1e-6
      @test norm(b_cpu - A_cpu * x) ≤ 1e-6

      A_gpu = CuSparseMatrixCSR(A_cpu[:,p])
      P = ilu02(A_gpu)
      function ldiv_ilu0!(P::CuSparseMatrixCSR, x, y, z)
        ldiv!(z, UnitLowerTriangular(P), x)
        ldiv!(y, UpperTriangular(P), z)
        return y
      end
      opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P, x, y, z))
      x̄, stats = gmres(A_gpu, b_gpu, M=opM)
      x = Vector(x̄)[invp]
      @test norm(b_gpu - A_gpu * x̄) ≤ 1e-6
      @test norm(b_cpu - A_cpu * x) ≤ 1e-6
    end
  end

  for FC in (Float32, Float64, ComplexF32, ComplexF64)
    S = CuVector{FC}
    V = CuSparseVector{FC}
    M = CuMatrix{FC}
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
      A, b = nonsymmetric_indefinite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = gmres(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "CG -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = cg(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "MINRES-QLP -- $FC" begin
      A, b = symmetric_indefinite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = minres_qlp(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "processes -- $FC" begin
      test_processes(S, M)
    end

    @testset "solver -- $FC" begin
      test_solver(S, M)
    end

    @testset "ktypeof -- $FC" begin
      dv = S(rand(FC, 10))
      b = view(dv, 4:8)
      @test Krylov.ktypeof(dv) <: S
      @test Krylov.ktypeof(b)  <: S

      dm = M(rand(FC, 10, 10))
      b = view(dm, :, 3)
      @test Krylov.ktypeof(b) <: S

      sv = V(sprand(FC, 10, 0.5))
      b = view(sv, 4:8)
      @test Krylov.ktypeof(sv) <: S
      @test Krylov.ktypeof(b)  <: S
    end
  end
end
