@testset "diom" begin
  diom_tol = 1.0e-6
  
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = diom(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = diom(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = diom(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = diom(A, b, memory=150)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = diom(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = diom(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ diom_tol)
      @test(stats.solved)

      # test callback function
      workspace = DiomWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      diom!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      # trust region tests, same as in test_cg
      # Test radius > 0  and b^TAb=0
      A, b = zero_rhs(FC=FC)
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b,radius = 10 * real(one(FC)))
      x, stats = solver.x, solver.stats
      @test stats.status == "x is a zero-residual solution"
      @test norm(x) == zero(FC)
      @test stats.niter == 0

      # Test radius > 0 and pᵀAp < 0
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b; radius = 10 * real(one(FC)))
      x, stats, = solver.x, solver.stats
      @test stats.indefinite == true
      
      # Test residual of the solution with trust region
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
      x, stats, = solver.x, solver.stats
      r = b - A * x
      normr = norm(r)
      @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
      @test stats.status == "on trust-region boundary"

      # test quadratic function values are computed correctly
      A = FC[10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b; radius = 10 * real(one(FC)), history = true)
      x, stats, = solver.x, solver.stats
      qxs = stats.qvals
      q = -dot(b, x) + dot(x, A * x)/2
      @test length(qxs) == stats.niter + 1
      @test abs(qxs[end] - q) ≤ 1.0e-10
      @test abs(qxs[1]) ≤ 1.0e-10  # q(0) = 0
      # test that q is decreasing
      @test all(diff(qxs) .<= 1.0e-10)

      # test quadratic function with trust-region
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
      x, stats, = solver.x, solver.stats
      q = -dot(b, x) + dot(x, A * x)/2
      qxs = stats.qvals
      @test abs(q - qxs[end]) ≤ 1.0e-10
      @test stats.status == "on trust-region boundary"

      # test trust-region with warm start
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      x0 = FC[0.5, 0.5, 0.5, 0.05]
      solver = DiomWorkspace(A, b)
      diom!(solver, A, b, x0; radius = 0.5 * real(one(FC)), history = true)
      x, stats, = solver.x, solver.stats
      q = -dot(b, x0) + dot(x0, A * x0)/2
      qxs = stats.qvals
      @test abs(q - qxs[1]) ≤ 1.0e-10
      r = b - A * x
      normr = norm(r)
      @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
      @test stats.status == "on trust-region boundary"

      # test split preconditioning with trust-region
      function LinearAlgebra.ldiv!(y::AbstractVector, M::SparseMatrixCSC, x::AbstractVector,)
        y .= M \ x
        return y
      end
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = diom(A, b, M=M, N=M, radius = 0.1 * real(one(FC)), history = true)
      r = b - A * x
      normr = norm(M * r)
      @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
      @test stats.status == "on trust-region boundary"
      q = -dot(b, x) + dot(x, A * x)/2
      qxs = stats.qvals
      @test abs(q - qxs[end]) ≤ 1.0e-10
    
      @test_throws TypeError diom(A, b, callback = workspace -> "string", history = true)
    end
  end
end
