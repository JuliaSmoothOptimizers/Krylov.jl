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

      Krylov.kdot(n, x, y)

      Krylov.kdotr(n, x, y)

      Krylov.knorm(n, x)

      Krylov.kaxpy!(n, a, x, y)
      Krylov.kaxpy!(n, a2, x, y)

      Krylov.kaxpby!(n, a, x, b, y)
      Krylov.kaxpby!(n, a2, x, b, y)
      Krylov.kaxpby!(n, a, x, b2, y)
      Krylov.kaxpby!(n, a2, x, b2, y)

      Krylov.kcopy!(n, y, x)

      Krylov.kfill!(x, a)

      Krylov.@kswap!(x, y)

      Krylov.kref!(n, x, y, c, s)
    end
  end

  @testset "reduced QR (larfg! / geqrf! / orgqr! / ormqr!)" begin
    @testset "accuracy $FC" for FC in (Float32, Float64, ComplexF32, ComplexF64, Complex{BigFloat}, BigFloat)
      T = real(FC)
      tol = 100 * eps(T)
      for (m, k) in ((10, 4), (6, 6))
        A = rand(FC, m, k)

        # larfg!: Hᴴ [α; x] = [β; 0] with v = [1; tail]  (LAPACK convention)
        α = A[1,1]
        x = A[2:m, 1]
        xold = copy(x)
        β, τ = Krylov.larfg!(α, x)
        v = vcat(one(FC), x)
        u = vcat(α, xold)
        Hᴴu = u - conj(τ) * v * (v' * u)
        @test abs(β) ≈ norm(u) atol=tol
        @test Hᴴu[1] ≈ β atol=tol
        @test norm(Hᴴu[2:end]) ≤ tol

        # geqrf! + orgqr!: reduced QR factorization
        QR = copy(A)
        tau = zeros(FC, k)
        Krylov.geqr2!(QR, tau)
        R = triu(QR[1:k, 1:k])
        Krylov.ung2r!(QR, tau)
        Q = QR[:, 1:k]
        @test istriu(R)
        @test norm(Q' * Q - I) ≤ tol
        @test norm(Q * R - A) ≤ tol * norm(A)

        Aref = copy(A)
        tauref = zeros(FC, k)
        Krylov.geqr2!(Aref, tauref)
        Qfull = zeros(FC, m, m)
        Qfull[:, 1:k] .= Aref
        Krylov.ung2r!(Qfull, tauref)
        nc = 5
        for trans in (FC <: Complex ? ('N', 'C') : ('N', 'T', 'C'))
          op = trans == 'N' ? Qfull : Qfull'
          C = rand(FC, m, nc)
          @test norm(Krylov.unm2r!('L', trans, copy(Aref), tauref, copy(C)) - op * C) ≤ tol
          D = rand(FC, nc, m)
          @test norm(Krylov.unm2r!('R', trans, copy(Aref), tauref, copy(D)) - D * op) ≤ tol
        end
      end
    end

    @testset "matches LAPACK $FC" for FC in (Float32, Float64, ComplexF32, ComplexF64)
      m, k = 9, 4
      A = rand(FC, m, k)
      Qg = copy(A); tg = zeros(FC, k); Krylov.geqr2!(Qg, tg);  Krylov.ung2r!(Qg, tg)   # pure Julia
      Ql = copy(A); tl = zeros(FC, k); Krylov.kgeqrf!(Ql, tl); Krylov.kungqr!(Ql, tl)  # LAPACK
      # Q is defined up to a phase, so compare the (phase-invariant) projectors
      @test Qg[:, 1:k] * Qg[:, 1:k]' ≈ Ql[:, 1:k] * Ql[:, 1:k]'
    end

    @testset "no allocations $FC" for FC in (Float16, ComplexF16)
      m, k, nc = 12, 4, 5

      A = rand(FC, m, k); tau = zeros(FC, k)
      Krylov.geqr2!(copy(A), copy(tau))
      Ag = copy(A)
      taug = copy(tau)
      @test (@allocated Krylov.geqr2!(Ag, taug)) == 0

      Af = copy(A)
      tf = zeros(FC, k)
      Krylov.geqr2!(Af, tf)
      Krylov.ung2r!(copy(Af), tf)
      Ao = copy(Af)
      @test (@allocated Krylov.ung2r!(Ao, tf)) == 0

      CL = rand(FC, m, nc)   # left operand
      CR = rand(FC, nc, m)   # right operand
      Krylov.unm2r!('L', 'N', Af, tf, copy(CL))
      for trans in (FC <: Complex ? ('N', 'C') : ('N', 'T', 'C'))
        DL = copy(CL)
        @test (@allocated Krylov.unm2r!('L', trans, Af, tf, DL)) == 0
        DR = copy(CR)
        @test (@allocated Krylov.unm2r!('R', trans, Af, tf, DR)) == 0
      end

      if VERSION ≥ v"1.12"
        α = rand(FC); x = rand(FC, m-1)
        Krylov.larfg!(α, copy(x))
        xl = copy(x)
        @test (@allocated Krylov.larfg!(α, xl)) == 0
      end
    end
  end
end
