@testset "bicgstab" begin
  bicgstab_tol = 1.0e-6

    for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = bicgstab(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = bicgstab(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Left preconditioning
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = bicgstab(A, b, M=M)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = bicgstab(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(500, 32)
      (x, stats) = bicgstab(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ bicgstab_tol)
      @test(stats.solved)

      # Test bᵀc == 0
      A, b, c = bc_breakdown(FC=FC)
      (x, stats) = bicgstab(A, b, c=c)
      @test stats.status == "Breakdown bᵀc = 0"

      # test callback function
      solver = BicgstabSolver(A, b)
      storage_vec = similar(b, size(A, 1))
      tol = 1.0e-1
      bicgstab!(solver, A, b,
              callback = (args...) -> test_callback_n2(args..., storage_vec = storage_vec, tol = tol))
      @test solver.stats.status == "user-requested exit"
      @test test_callback_n2(solver, A, b, storage_vec = storage_vec, tol = tol)

      @test_throws TypeError bicgstab(A, b, callback = (args...) -> "string", history = true)
    end
  end
end
