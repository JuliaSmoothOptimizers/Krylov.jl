@testset "usymqr" begin
  usymqr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      c = copy(b)
      (x, stats) = usymqr(A, b, c)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Underdetermined and consistent systems.
      A, b = under_consistent(FC=FC)
      c = ones(FC, 25)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)

      # Underdetermined and inconsistent systems.
      A, b = under_inconsistent(FC=FC)
      c = [iseven(i) ? one(FC) : -one(FC) for i=1:25]
      (x, stats) = usymqr(A, b, c)
      @test stats.inconsistent

      # Square and consistent systems.
      A, b = square_consistent(FC=FC)
      c = ones(FC, 10)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)

      # Square and inconsistent systems.
      A, b = square_inconsistent(FC=FC)
      c = ones(FC, 10)
      (x, stats) = usymqr(A, b, c)
      @test stats.inconsistent

      # Overdetermined and consistent systems.
      A, b = over_consistent(FC=FC)
      c = ones(FC, 10)
      (x, stats) = usymqr(A, b, c)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)

      # Overdetermined and inconsistent systems.
      A, b = over_inconsistent(FC=FC)
      c = [2^i * (iseven(i) ? one(FC) : -one(FC)) for i=1:10]
      (x, stats) = usymqr(A, b, c)
      @test stats.inconsistent

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = usymqr(A, b, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ usymqr_tol)
      @test(stats.solved)
    end
  end
end
