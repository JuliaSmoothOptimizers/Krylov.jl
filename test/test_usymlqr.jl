@testset "usymlqr" begin
  usymlqr_tol = 1.0e-6

  @testset "Data Type: $FC" for FC in (Float64, ComplexF64)
    # Test saddle-point systems
    A, b, D = saddle_point(FC=FC)
    m, n = size(A)
    c = -b
    D⁻¹ = sparse(inv(D))
    N⁻¹ = eye(n)
    H⁻¹ = blockdiag(D⁻¹, N⁻¹)

    (x, y, stats) = usymlqr(A, b, c, ls=true, ln=false)
    K = [eye(m) A; A' zeros(n, n)]
    d = [b; 0*c]
    r =  d - K * [x; y]
    resid = norm(r) / norm(d)
    @test(resid ≤ usymlqr_tol)

    (x, y, stats) = usymlqr(A, b, c, ls=false, ln=true)
    K = [eye(m) A; A' zeros(n, n)]
    d = [0*b; c]
    r =  d - K * [x; y]
    resid = norm(r) / norm(d)
    @test(resid ≤ usymlqr_tol)

    (x, y, stats) = usymlqr(A, b, c)
    K = [eye(m) A; A' zeros(n, n)]
    d = [b; c]
    r =  d - K * [x; y]
    resid = norm(r) / norm(d)
    @test(resid ≤ usymlqr_tol)

    # Test different types for input and output
    d = TestVector(c)
    workspace = UsymlqrWorkspace(KrylovConstructor(b, d))
    usymlqr!(workspace, A, b, d)
    @test typeof(workspace.x) === typeof(b)
    @test typeof(workspace.y) === typeof(d)
    @test workspace.stats.solved

    # (x, y, stats) = usymlqr(A, b, c, M=D⁻¹, ls=true, ln=false)
    # K = [D A; A' zeros(n, n)]
    # d = [b; zeros(FC, n)]
    # r =  d - K * [x; y]
    # resid = sqrt(dot(r, H⁻¹ * r) |> real) / sqrt(dot(d, H⁻¹ * d) |> real)
    # @printf("USYMLQR: Relative residual: %8.1e\n", resid)
    # @test(resid ≤ usymlqr_tol)

    # (x, y, stats) = usymlqr(A, b, c, M=D⁻¹, ls=false, ln=true)
    # K = [D A; A' zeros(n, n)]
    # d = [zeros(FC, m); c]
    # r =  d - K * [x; y]
    # resid = sqrt(dot(r, H⁻¹ * r) |> real) / sqrt(dot(d, H⁻¹ * d) |> real)
    # @printf("USYMLQR: Relative residual: %8.1e\n", resid)
    # @test(resid ≤ usymlqr_tol)

    # (x, y, stats) = usymlqr(A, b, c, M=D⁻¹)
    # K = [D A; A' zeros(n, n)]
    # d = [b; c]
    # r =  d - K * [x; y]
    # resid = sqrt(dot(r, H⁻¹ * r) |> real) / sqrt(dot(d, H⁻¹ * d) |> real)
    # @printf("USYMLQR: Relative residual: %8.1e\n", resid)
    # @test(resid ≤ usymlqr_tol)
  end
end
