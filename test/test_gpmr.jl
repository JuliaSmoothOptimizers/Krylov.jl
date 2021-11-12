@testset "gpmr" begin
  gpmr_tol = 1.0e-6

  for transpose ∈ (false, true)
    A, B, b, c, M, N = gsp(transpose)
    K = [I A; B I]
    d = [b; c]

    # Test reorthogonalization
    x, y, stats = gpmr(A, B, b, c, reorthogonalization=true)
    r =  d - K * [x; y]
    resid = norm(r) / norm(d)
    @test(resid ≤ gpmr_tol)

    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    H⁻¹ = blockdiag(sparse(M⁻¹), sparse(N⁻¹))
    P⁻¹ = blockdiag(sparse(sqrt(M⁻¹)), sparse(sqrt(N⁻¹)))
    K = [M A; B N]

    # Test left preconditioning
    x, y, stats = gpmr(A, B, b, c, C=M⁻¹, D=N⁻¹)
    r =  d - K * [x; y]
    resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
    @test(resid ≤ gpmr_tol)

    # Test right preconditioning
    x, y, stats = gpmr(A, B, b, c, E=M⁻¹, F=N⁻¹)
    res = norm(d - K * [x; y])
    resid = norm(r) / norm(d)
    @test(resid ≤ gpmr_tol)

    # Test split preconditioning
    x, y, stats = gpmr(A, B, b, c, C=sqrt(M⁻¹), D=sqrt(N⁻¹), E=sqrt(M⁻¹), F=sqrt(N⁻¹))
    res = norm(d - K * [x; y])
    resid = norm(P⁻¹ * r) / norm(P⁻¹ * d)
    @test(resid ≤ gpmr_tol)
  end

  # Test underdetermined adjoint systems.
  A, b, c = underdetermined_adjoint()
  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=0.0)
  r = b - A  * y
  s = c - A' * x
  resid = norm([r; s]) / norm([b; c])
  @test(resid ≤ gpmr_tol)

  # Test square adjoint systems.
  A, b, c = square_adjoint()
  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=0.0)
  r = b - A  * y
  s = c - A' * x
  resid = norm([r; s]) / norm([b; c])
  @test(resid ≤ gpmr_tol)

  # Test overdetermined adjoint systems
  A, b, c = overdetermined_adjoint()
  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=0.0)
  r = b - A  * y
  s = c - A' * x
  resid = norm([r; s]) / norm([b; c])
  @test(resid ≤ gpmr_tol)

  # Test adjoint ODEs.
  A, b, c = adjoint_ode()
  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=0.0)
  r = b - A  * y
  s = c - A' * x
  resid = norm([r; s]) / norm([b; c])
  @test(resid ≤ gpmr_tol)

  # Test adjoint PDEs.
  A, b, c = adjoint_pde()
  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=0.0)
  r = b - A  * y
  s = c - A' * x
  resid = norm([r; s]) / norm([b; c])
  @test(resid ≤ gpmr_tol)

  # Test saddle-point systems
  A, b, D = saddle_point()
  m, n = size(A)
  c = -b
  D⁻¹ = sparse(inv(D))
  N⁻¹ = eye(n)
  H⁻¹ = blockdiag(D⁻¹, N⁻¹)
  G⁻¹ = blockdiag(N⁻¹, D⁻¹)

  (x, y, stats) = gpmr(A, A', b, c, gsp=true, E=D⁻¹)
  K = [D A; A' spzeros(n, n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, gsp=true, C=D⁻¹)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, gsp=true)
  K = [eye(m) A; A' spzeros(n, n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

    (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=1.0, F=D⁻¹)
  K = [spzeros(m, m) A; A' D]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=1.0, D=D⁻¹)
  r =  d - K * [x; y]
  resid = norm(G⁻¹ * r) / norm(G⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=0.0, μ=1.0)
  K = [spzeros(m, m) A; A' eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  # Test symmetric and quasi-definite systems
  A, b, M, N = sqd()
  m, n = size(A)
  c = -b
  M⁻¹ = sparse(inv(M))
  N⁻¹ = sparse(inv(N))
  H⁻¹ = blockdiag(M⁻¹, N⁻¹)

  (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹, λ=1.0, μ=-1.0)
  K = [M A; A' -N]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹, λ=1.0, μ=-1.0)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=1.0, μ=-1.0)
  K = [eye(m) A; A' -eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  λ = 12.0; μ =-0.7
  (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹, λ=λ, μ=μ)
  K = [λ*M A; A' μ*N]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹, λ=λ, μ=μ)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=λ, μ=μ)
  K = [λ*eye(m) A; A' μ*eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  λ = -1e-6; μ =1e-8
  (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹, λ=λ, μ=μ)
  K = [λ*M A; A' μ*N]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹, λ=λ, μ=μ)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=λ, μ=μ)
  K = [λ*eye(m) A; A' μ*eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  # Test symmetric positive definite systems
  (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹)
  K = [M A; A' N]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c)
  K = [eye(m) A; A' eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  # Test symmetric negative definite systems
  (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹, λ=-1.0, μ=-1.0)
  K = [-M A; A' -N]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹, λ=-1.0, μ=-1.0)
  r =  d - K * [x; y]
  resid = norm(H⁻¹ * r) / norm(H⁻¹ * d)
  @test(resid ≤ gpmr_tol)

  (x, y, stats) = gpmr(A, A', b, c, λ=-1.0, μ=-1.0)
  K = [-eye(m) A; A' -eye(n)]
  d = [b; c]
  r =  d - K * [x; y]
  resid = norm(r) / norm(d)
  @test(resid ≤ gpmr_tol)

  # Test dimension of additional vectors
  for transpose ∈ (false, true)
    A, b, c, M, N = small_sqd(transpose)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    (x, y, stats) = gpmr(A, A', b, c, E=M⁻¹, F=N⁻¹)
    (x, y, stats) = gpmr(A, A', b, c, C=M⁻¹, D=N⁻¹)
  end

  # Test restart
  A, b = restart()
  solver = GpmrSolver(A, b, 20)
  gpmr!(solver, A, A', b, b, itmax=20)
  @test !issolved(solver)
  gpmr!(solver, A, A', b, b, restart=true)
  r = [b - solver.x - A * solver.y; b - A' * solver.x - solver.y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ gpmr_tol)
  @test issolved(solver)
end
