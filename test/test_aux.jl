@testset "aux" begin
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

  # test roots of a quadratic
  roots = Krylov.roots_quadratic(0.0, 0.0, 0.0)
  @test length(roots) == 1
  @test roots[1] == 0.0

  roots = Krylov.roots_quadratic(0.0, 0.0, 1.0)
  @test length(roots) == 0

  roots = Krylov.roots_quadratic(0.0, 3.14, -1.0)
  @test length(roots) == 1
  @test roots[1] == 1.0 / 3.14

  roots = Krylov.roots_quadratic(1.0, 0.0, 1.0)
  @test length(roots) == 0

  roots = Krylov.roots_quadratic(1.0, 0.0, 0.0)
  @test length(roots) == 2
  @test roots[1] == 0.0
  @test roots[2] == 0.0

  roots = Krylov.roots_quadratic(1.0, 3.0, 2.0)
  @test length(roots) == 2
  @test roots[1] ≈ -2.0
  @test roots[2] ≈ -1.0

  roots = Krylov.roots_quadratic(1.0e+8, 1.0, 1.0)
  @test length(roots) == 0

  # ill-conditioned quadratic
  roots = Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=0)
  @test length(roots) == 2
  @test roots[1] == 1.0e+13
  @test roots[2] == 0.0

  # iterative refinement is crucial!
  roots = Krylov.roots_quadratic(-1.0e-8, 1.0e+5, 1.0, nitref=1)
  @test length(roots) == 2
  @test roots[1] == 1.0e+13
  @test roots[2] == -1.0e-05

  # not ill-conditioned quadratic
  roots = Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=0)
  @test length(roots) == 2
  @test isapprox(roots[1],  1.0e+7, rtol=1.0e-6)
  @test isapprox(roots[2], -1.0, rtol=1.0e-6)

  roots = Krylov.roots_quadratic(-1.0e-7, 1.0, 1.0, nitref=1)
  @test length(roots) == 2
  @test isapprox(roots[1], 1.0e+7, rtol=1.0e-6)
  @test isapprox(roots[2], -1.0, rtol=1.0e-6)

  # test trust-region boundary
  x = ones(5)
  d = ones(5); d[1:2:5] .= -1
  @test_throws ErrorException Krylov.to_boundary(x, d, -1.0)
  @test_throws ErrorException Krylov.to_boundary(x, d, 0.5)
  @test_throws ErrorException Krylov.to_boundary(x, zeros(5), 1.0)
  @test maximum(Krylov.to_boundary(x, d, 5.0)) ≈ 2.209975124224178
  @test minimum(Krylov.to_boundary(x, d, 5.0)) ≈ -1.8099751242241782
  @test maximum(Krylov.to_boundary(x, d, 5.0, flip=true)) ≈ 1.8099751242241782
  @test minimum(Krylov.to_boundary(x, d, 5.0, flip=true)) ≈ -2.209975124224178

  # test kzeros and kones
  @test Krylov.kzeros(Vector{Float64}, 10) == zeros(10)
  @test Krylov.kones(Vector{Float64}, 10) == ones(10)

  # test ktypeof
  a = sprand(Float32, 10, 0.5)
  b = view(a, 4:8)
  @test Krylov.ktypeof(a) == Vector{Float32}
  @test Krylov.ktypeof(b) == Vector{Float32}

  a = sprand(Float64, 10, 0.5)
  b = view(a, 4:8)
  @test Krylov.ktypeof(a) == Vector{Float64}
  @test Krylov.ktypeof(b) == Vector{Float64}
end
