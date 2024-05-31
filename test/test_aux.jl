@testset "aux" begin

  @testset "sym_givens" begin
    # test Givens reflector corner cases
    (c, s, ρ) = Krylov.sym_givens(0.0, 0.0)
    @test (c == 1.0) && (s == 0.0) && (ρ == 0.0)

    a = 3.14
    (c, s, ρ) = Krylov.sym_givens(a, 0.0)
    @test (c == 1.0) && (s == 0.0) && (ρ == a)
    (c, s, ρ) = Krylov.sym_givens(-a, 0.0)
    @test (c == -1.0) && (s == 0.0) && (ρ == a)

    b = 3.14
    (c, s, ρ) = Krylov.sym_givens(0.0, b)
    @test (c == 0.0) && (s == 1.0) && (ρ == b)
    (c, s, ρ) = Krylov.sym_givens(0.0, -b)
    @test (c == 0.0) && (s == -1.0) && (ρ == b)

    (c, s, ρ) = Krylov.sym_givens(Complex(0.0), Complex(0.0))
    @test (c == 1.0) && (s == Complex(0.0)) && (ρ == Complex(0.0))

    a = Complex(1.0, 1.0)
    (c, s, ρ) = Krylov.sym_givens(a, Complex(0.0))
    @test (c == 1.0) && (s == Complex(0.0)) && (ρ == a)
    (c, s, ρ) = Krylov.sym_givens(-a, Complex(0.0))
    @test (c == 1.0) && (s == Complex(0.0)) && (ρ == -a)

    b = Complex(1.0, 1.0)
    (c, s, ρ) = Krylov.sym_givens(Complex(0.0), b)
    @test (c == 0.0) && (s == Complex(1.0)) && (ρ == b)
    (c, s, ρ) = Krylov.sym_givens(Complex(0.0), -b)
    @test (c == 0.0) && (s == Complex(1.0)) && (ρ == -b)
  end

  @testset "roots_quadratic" begin
    # test roots of a quadratic
    roots = Krylov.roots_quadratic(0.0, 0.0, 0.0)
    @test roots[1] == 0.0
    @test roots[2] == 0.0

    @test_throws ErrorException Krylov.roots_quadratic(0.0, 0.0, 1.0)

    roots = Krylov.roots_quadratic(0.0, 3.14, -1.0)
    @test roots[1] == 1.0 / 3.14
    @test roots[2] == 1.0 / 3.14

    @test_throws ErrorException Krylov.roots_quadratic(1.0, 0.0, 1.0)

    roots = Krylov.roots_quadratic(1.0, 0.0, 0.0)
    @test roots[1] == 0.0
    @test roots[2] == 0.0

    roots = Krylov.roots_quadratic(1.0, 3.0, 2.0)
    @test roots[1] ≈ -2.0
    @test roots[2] ≈ -1.0

    @test_throws ErrorException Krylov.roots_quadratic(1.0e+8, 1.0, 1.0)

    # ill-conditioned quadratic
    roots = Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=0)
    @test roots[1] == 1.0e+13
    @test roots[2] == 0.0

    # iterative refinement is crucial!
    roots = Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=1)
    @test roots[1] == 1.0e+13
    @test roots[2] == -1.0e-05

    # not ill-conditioned quadratic
    roots = Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=0)
    @test isapprox(roots[1],  1.0e+7, rtol=1.0e-6)
    @test isapprox(roots[2], -1.0, rtol=1.0e-6)

    roots = Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=1)
    @test isapprox(roots[1], 1.0e+7, rtol=1.0e-6)
    @test isapprox(roots[2], -1.0, rtol=1.0e-6)

    allocations = @allocated Krylov.roots_quadratic(0.0, 0.0, 0.0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(0.0, 3.14, -1.0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(1.0, 0.0, 0.0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(1.0, 3.0, 2.0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=1)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=0)
    @test allocations == 0

    allocations = @allocated Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=1)
    @test allocations == 0
  end

  @testset "to_boundary" begin
    # test trust-region boundary
    n = 5
    x = ones(n)
    d = ones(n); d[1:2:n] .= -1
    z = similar(d) # <-- placeholder for preconditioning storage
    @test_throws ErrorException Krylov.to_boundary(n, x, d, z, -1.0)
    @test_throws ErrorException Krylov.to_boundary(n, x, d, z, 0.5)
    @test_throws ErrorException Krylov.to_boundary(n, x, zeros(n), z, 1.0)
    @test maximum(Krylov.to_boundary(n, x, d, z, 5.0)) ≈ 2.209975124224178
    @test minimum(Krylov.to_boundary(n, x, d, z, 5.0)) ≈ -1.8099751242241782
    @test maximum(Krylov.to_boundary(n, x, d, z, 5.0, flip=true)) ≈ 1.8099751242241782
    @test minimum(Krylov.to_boundary(n, x, d, z, 5.0, flip=true)) ≈ -2.209975124224178
  end

  @testset "kzeros" begin
    # test kzeros
    @test Krylov.kzeros(Vector{Float64}, 10) == zeros(Float64, 10)
    @test Krylov.kzeros(Vector{ComplexF32}, 10) == zeros(ComplexF32, 10)
  end

  @testset "kones" begin
    # test kones
    @test Krylov.kones(Vector{Float64}, 10) == ones(Float64, 10)
    @test Krylov.kones(Vector{ComplexF32}, 10) == ones(ComplexF32, 10)
  end

  @testset "ktypeof" begin
    # test ktypeof
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
      dv = rand(FC, 10)
      b = view(dv, 4:8)
      @test Krylov.ktypeof(dv) == Vector{FC}
      @test Krylov.ktypeof(b)  == Vector{FC}

      dm = rand(FC, 10, 10)
      b = view(dm, :, 3)
      @test Krylov.ktypeof(b) == Vector{FC}

      sv = sprand(FC, 10, 0.5)
      b = view(sv, 4:8)
      @test Krylov.ktypeof(sv) == Vector{FC}
      @test Krylov.ktypeof(b)  == Vector{FC}
    end
  end

  @testset "vector_to_matrix" begin
    # test vector_to_matrix
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
      S = Vector{FC}
      M = Krylov.vector_to_matrix(S)
      @test M == Matrix{FC}
    end
  end

  @testset "matrix_to_vector" begin
    # test matrix_to_vector
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
      M = Matrix{FC}
      S = Krylov.matrix_to_vector(M)
      @test S == Vector{FC}
    end
  end

  @testset "macros" begin
    # test macros
    for FC ∈ (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
      n = 10
      x = rand(FC, n)
      y = rand(FC, n)
      a = rand(FC)
      b = rand(FC)
      c = rand(FC)
      s = rand(FC)

      T = real(FC)
      a2 = rand(T)
      b2 = rand(T)

      Krylov.@kdot(n, x, y)

      Krylov.@kdotr(n, x, y)

      Krylov.@knrm2(n, x)

      Krylov.@kaxpy!(n, a, x, y)
      Krylov.@kaxpy!(n, a2, x, y)

      Krylov.@kaxpby!(n, a, x, b, y)
      Krylov.@kaxpby!(n, a2, x, b, y)
      Krylov.@kaxpby!(n, a, x, b2, y)
      Krylov.@kaxpby!(n, a2, x, b2, y)

      Krylov.@kcopy!(n, x, y)

      Krylov.@kswap(x, y)

      Krylov.@kref!(n, x, y, c, s)
    end
  end
end
