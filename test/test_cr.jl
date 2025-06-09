@testset "cr" begin
    cr_tol = 1.0e-6

    for FC in (Float64, ComplexF64)
        @testset "Data Type: $FC" begin
            # Symmetric and positive definite system.
            A, b = symmetric_definite(FC = FC)
            (x, stats) = cr(A, b)
            r = b - A * x
            resid = norm(r) / norm(b)
            @test(resid ≤ cr_tol)
            @test(stats.solved)
            @test stats.indefinite == false

            # Code coverage
            (x, stats) = cr(Matrix(A), b)

            if FC == Float64
                radius = 0.75 * norm(x)
                (x, stats) = cr(A, b, radius = radius)
                @test(stats.solved)
                @test abs(norm(x) - radius) ≤ cr_tol * radius

                # Sparse Laplacian
                A, _ = sparse_laplacian(FC = FC)
                b = randn(size(A, 1))
                itmax = 0
                # case: ‖x*‖ > Δ
                radius = 10.0
                (x, stats) = cr(A, b, radius = radius)
                xNorm = norm(x)
                r = b - A * x
                resid = norm(r) / norm(b)
                @test abs(xNorm - radius) ≤ cr_tol * radius
                @test(stats.solved)
                # case: ‖x*‖ < Δ
                radius = 30.0
                (x, stats) = cr(A, b, radius = radius)
                xNorm = norm(x)
                r = b - A * x
                resid = norm(r) / norm(b)
                @test(resid ≤ cr_tol)
                @test(stats.solved)

                radius = 0.75 * xNorm
                (x, stats) = cr(A, b, radius = radius)
                @test(stats.solved)
                @test(abs(radius - norm(x)) ≤ cr_tol * radius)
            end

            # Test b == 0
            A, b = zero_rhs(FC = FC)
            (x, stats) = cr(A, b)
            @test norm(x) == 0
            @test stats.status == "x is a zero-residual solution"

            # Test with Jacobi (or diagonal) preconditioner
            A, b, M = square_preconditioned(FC = FC)
            (x, stats) = cr(A, b, M = M, atol = 1e-5, rtol = 0.0)
            r = b - A * x
            resid = sqrt(real(dot(r, M * r))) / sqrt(real(dot(b, M * b)))
            @test(resid ≤ 10 * cr_tol)
            @test(stats.solved)

            # Test linesearch
            # Iter=0: bᵀ Ab = 0 → zero-curvature at k=0
            A, b = system_zero_quad(FC = Float64)   # ensures bᵀ A b == 0
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; linesearch = true)
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.niter == 0
            @test stats.status == "b is a zero-curvature direction"
            @test real(dot(npc_dir, A * npc_dir)) ≈ 0

            # Test Linesearch which would stop on the first call since A is indefinite
            A, b = symmetric_indefinite(FC = FC; shift = 10)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b, linesearch = true)
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.status == "nonpositive curvature"
            @test stats.niter == 0
            @test stats.solved == true
            @test stats.indefinite == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0
            @test all(x .== b)

            # Test when b^TAb=0 and linesearch is false
            A, b = system_zero_quad(FC = FC)
            x, stats = cr(A, b, linesearch = false)
            @test stats.status == "b is a zero-curvature direction"
            @test norm(x) == zero(FC)
            @test stats.solved == true
            @test stats.niter == 0

            # test callback function
            A, b = symmetric_definite(FC = FC)
            workspace = CrWorkspace(A, b)
            tol = 1.0e-1
            cb_n2 = TestCallbackN2(A, b, tol = tol)
            cr!(workspace, A, b, callback = cb_n2)
            @test workspace.stats.status == "user-requested exit"
            @test cb_n2(workspace)

            @test_throws TypeError cr(
                A,
                b,
                callback = workspace -> "string",
                history = true,
            )

            # 2 negative curvature
            A = FC[
                1.0 0.0;
                0.0 0.0
            ]
            b = ones(FC, 2)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; linesearch = true)
            x, stats, npc_dir, p = solver.x, solver.stats, solver.npc_dir, solver.p
            @test stats.npcCount == 2
            @test real(dot(npc_dir, A*npc_dir)) ≤ norm(npc_dir)^2 + cr_tol
            @test real(dot(p, A*p)) < cr_tol

            # Only -p negative curvature
            A = FC(-1.0)*I(2)
            b = ones(FC, 2)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; linesearch = true)
            x, stats, npc_dir, p = solver.x, solver.stats, solver.npc_dir, solver.p
            @test stats.status == "nonpositive curvature"
            @test stats.npcCount == 1
            @test real(dot(npc_dir, A*npc_dir)) ≤ cr_tol

            @test real(dot(p, A*p)) < 0

            # Warm-start + linesearch must error
            A, b = symmetric_indefinite(FC = Float64)
            @test_throws MethodError cr(A, b; warm_start = true, linesearch = true)

            # radius > 0
            # bᵀ Ab ≈  0 at first iteration
            A = FC[
                1.0 1e-10;
                1e-10 1e-10
            ]
            b = FC[0.0, 0.99]
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)), γ = 10 * real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir

            @test stats.npcCount == 1
            @test stats.status == "on trust-region boundary"
            @test stats.indefinite == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0.01 # almost zero curvature
            @test stats.solved == true


            # pᵀAp < 0 and rᵀAr > 0 after the second iteration
            A = FC[
                10.0 0.0 0.0 0.0;
                0.0 8.0 0.0 0.0;
                0.0 0.0 5.0 0.0;
                0.0 0.0 0.0 -1.0
            ]
            b = FC[1.0, 1.0, 1.0, 0.1]
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)))
            x, stats, npc_dir, p = solver.x, solver.stats, solver.npc_dir, solver.p
            @test stats.npcCount == 2
            @test stats.status == "on trust-region boundary"
            @test stats.indefinite == true
            @test stats.solved == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0.01 # almost zero curvature
            @test real(dot(p, A * p)) < 0 # negative curvature


            # pᵀAp < 0 and rᵀAr > 0 after the second iteration
            A = FC[
                5.0 0.0 0.0;
                0.0 4.0 0.0;
                0.0 0.0 -1.0
            ]
            b = FC[1.0, 1.0, 0.1]
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)))
            x, stats, npc_dir, p = solver.x, solver.stats, solver.npc_dir, solver.p
            @test stats.npcCount == 2
            @test stats.status == "on trust-region boundary"
            @test stats.indefinite == true
            @test stats.solved == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0.01 # almost zero curvature
                  
            # pᵀ Ap ≈  0 after first iteration
            A = FC[
                1.0 1e-10;
                1e-10 1e-10
            ]
            b = FC[0.002, 1.0]
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)), γ = 10 * real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.npcCount == 1
            @test stats.solved == true
            @test stats.status == "on trust-region boundary"
            @test stats.indefinite == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0.01

            # Iter=0: bᵀ Ab = 0 → zero-curvature at k=0
            A, b = system_zero_quad(FC = Float64)   # ensures bᵀ A b == 0
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.niter == 0
            @test stats.status == "b is a zero-curvature direction"
            @test real(dot(npc_dir, A * npc_dir)) ≈ 0

            # Test stop since A is indefinite
            A, b = symmetric_indefinite(FC = FC; shift = 5)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b, radius = 10 * real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.status == "on trust-region boundary"
            @test stats.solved == true
            @test stats.indefinite == true
            @test real(dot(npc_dir, A * npc_dir)) <= 0
            @test stats.solved == true

            # negative curvature
            A = FC[
                1.0 0.0;
                0.0 0.0
            ]
            b = ones(FC, 2)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.npcCount == 1
            @test real(dot(npc_dir, A*npc_dir)) ≤ 0.0

            # p negative curvature
            A = FC(-1.0)*I(2)
            b = ones(FC, 2)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b; radius = 10 * real(one(FC)))
            x, stats, npc_dir, p = solver.x, solver.stats, solver.npc_dir, solver.p
            @test stats.status == "on trust-region boundary"
            @test stats.npcCount == 2
            @test real(dot(npc_dir, A*npc_dir)) ≤ cr_tol
            @test real(dot(p, A*p)) < 0


            # Test on trust-region boundary when radius > 0
            A, b = symmetric_indefinite(FC = FC, shift = 5)
            solver = CrWorkspace(A, b)
            cr!(solver, A, b, radius = real(one(FC)))
            x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
            @test stats.status == "on trust-region boundary"
            @test norm(x) ≈ 1.0
            @test real(dot(x, A * x)) <= 0

            # Test on trust-region boundary when radius = 1 and linesearch is true
            A, b = symmetric_indefinite(FC = FC, shift = 5)
            @test_throws ErrorException cr(A, b, radius = real(one(FC)), linesearch = true)

        end
    end
end
