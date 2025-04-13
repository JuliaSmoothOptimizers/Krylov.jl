@testset "cr" begin
  cr_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = cr(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ cr_tol)
      @test(stats.solved)

      # Code coverage
      (x, stats) = cr(Matrix(A), b)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = cr(A, b, radius=radius)
        @test(stats.solved)
        @test abs(norm(x) - radius) ≤ cr_tol * radius

        # Sparse Laplacian
        A, _ = sparse_laplacian(FC=FC)
        b = randn(size(A, 1))
        itmax = 0
        # case: ‖x*‖ > Δ
        radius = 10.
        (x, stats) = cr(A, b, radius=radius)
        xNorm = norm(x)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test abs(xNorm - radius) ≤ cr_tol * radius
        @test(stats.solved)
        # case: ‖x*‖ < Δ
        radius = 30.
        (x, stats) = cr(A, b, radius=radius)
        xNorm = norm(x)
        r = b - A * x
        resid = norm(r) / norm(b)
        @test(resid ≤ cr_tol)
        @test(stats.solved)

        radius = 0.75 * xNorm
        (x, stats) = cr(A, b, radius=radius)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ cr_tol * radius)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = cr(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = cr(A, b, M=M, atol=1e-5, rtol=0.0)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
      @test(resid ≤ 10 * cr_tol)
      @test(stats.solved)

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      x, stats = cr(A, b, linesearch=true)
      @test stats.status == "nonpositive curvature"

      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      x, stats = cr(A, b, linesearch=true)
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 0
      @test all(x .== b)
      @test stats.solved == true

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      x, stats = cr(A, b, linesearch=true)
      @test stats.status == "b is a zero-curvature direction"
      @test all(x .== b)
      @test stats.solved == true

      # Test when b^TAb=0 and linesearch is false
      A, b = system_zero_quad(FC=FC)
      x, stats = cr(A,b, linesearch=false)
      @test stats.status == "b is a zero-curvature direction"
      @test norm(x) == zero(FC)
      @test stats.solved == true

 
      # test callback function
      A, b = symmetric_definite(FC=FC)
      workspace = CrWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      cr!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError cr(A, b, callback = solver -> "string", history = true)
    end
  end
end
