function test_warm_start(FC)
  A, b = warm_start(FC=FC)
  n, m = size(A)
  x0 = 1.2 * ones(FC, n)
  y0 = 0.8 * ones(FC, n)
  tol = 1.0e-6

  x, y, stats = tricg(A, b, b, x0, y0)
  r = [b - x - A * y; b - A' * x + y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  x, y, stats = trimr(A, b, b, x0, y0)
  r = [b - x - A * y; b - A' * x + y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  x, y, stats = gpmr(A, A', b, b, x0, y0)
  r = [b - x - A * y; b - A' * x - y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  x, stats = minres_qlp(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  x, stats = symmlq(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  x, stats = cg(A, b, x0)
  r = b - A * x
  @test(resid ≤ tol)

  x, stats = minres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  x, stats = diom(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  x, stats = dqgmres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  x, stats = fom(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
      
  x, stats = gmres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
end

@testset "warm-start" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_warm_start(FC)
    end
  end
end
