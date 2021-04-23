@testset "bilqr" begin
  bilqr_tol = 1.0e-6

  # Test square adjoint systems.
  A, b, c = square_adjoint()
  (x, t, stats) = bilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ bilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ bilqr_tol)
  @test(stats.solved_dual)

  # Test adjoint ODEs.
  A, b, c = adjoint_ode()
  (x, t, stats) = bilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ bilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ bilqr_tol)
  @test(stats.solved_dual)

  # Test adjoint PDEs.
  A, b, c = adjoint_pde()
  (x, t, stats) = bilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ bilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ bilqr_tol)
  @test(stats.solved_dual)

  # Test bᵀc == 0
  A = rand(10, 10)
  b = Float64.([mod(i,2) for i = 1:10])
  c = Float64.([mod(i+1,2) for i = 1:10])
  (x, t, stats) = bilqr(A, b, c)
  @test stats.status == "Breakdown bᵀc = 0"
end
