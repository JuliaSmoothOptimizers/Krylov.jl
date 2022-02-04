@testset "cgls" begin
  cgls_tol = 1.0e-5

  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      for npower = 1 : 4
        (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

        (x, stats) = cgls(A, b)
        resid = norm(A' * (A*x - b)) / norm(b)
        @test(resid ≤ cgls_tol)
        @test(stats.solved)

        λ = 1.0e-3
        (x, stats) = cgls(A, b, λ=λ)
        resid = norm(A' * (A*x - b) + λ * x) / norm(b)
        @test(resid ≤ cgls_tol)
        @test(stats.solved)
      end

      # Test with preconditioning.
      A, b, M = saddle_point(FC=FC)
      M⁻¹ = inv(M)
      (x, stats) = cgls(A, b, M=M⁻¹)
      resid = norm(A' * M⁻¹ * (A * x - b)) / sqrt(dot(b, M⁻¹ * b))
      @test resid ≤ cgls_tol

      # test trust-region constraint
      (x, stats) = cgls(A, b)

      radius = 0.75 * norm(x)
      (x, stats) = cgls(A, b, radius=radius)
      @test(stats.solved)
      @test(abs(radius - norm(x)) ≤ cgls_tol * radius)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cgls(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, D = small_sp(transpose, FC=FC)
        D⁻¹ = inv(D)
        (x, stats) = cgls(A, b, M=D⁻¹, λ=1.0)
      end
    end
  end
end
