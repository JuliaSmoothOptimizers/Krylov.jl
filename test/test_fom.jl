include("get_div_grad.jl")

function test_fom()
  fom_tol = 1.0e-6
  
  for pivoting in (false, true)
    # Symmetric and positive definite systems (cubic spline matrix).
    n = 10
    A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
    b = A * [1:n;]
    (x, stats) = fom(A, b, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Symmetric indefinite variant.
    A = A - 3 * I
    b = A * [1:n;]
    (x, stats) = fom(A, b, itmax=10, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Nonsymmetric and positive definite systems.
    A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
    b = A * [1:n;]
    (x, stats) = fom(A, b, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Nonsymmetric indefinite variant.
    A = [i == j ? 10*(-1.0)^(i*j) : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
    b = A * [1:n;]
    (x, stats) = fom(A, b, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Code coverage.
    (x, stats) = fom(Matrix(A), b, pivoting=pivoting)
    show(stats)

    # Sparse Laplacian.
    A = get_div_grad(16, 16, 16)
    b = ones(size(A, 1))
    (x, stats) = fom(A, b, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Test with an user-defined x₀ (restarted FOM)
    x0 = ones(size(A, 1))
    (x, stats) = fom(A, b, x0=x0, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Symmetric indefinite variant, almost singular.
    # Also test reorthogonalization.
    A = A - 5 * I
    (x, stats) = fom(A, b, reorthogonalization=true, pivoting=pivoting)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Test b == 0
    (x, stats) = fom(A, zeros(size(A,1)), pivoting=pivoting)
    @test x == zeros(size(A,1))
    @test stats.status == "x = 0 is a zero-residual solution"

    # Test integer values
    A = spdiagm(-1 => ones(Int, n-1), 0 => 4*ones(Int, n), 1 => ones(Int, n-1))
    b = A * [1:n;]
    (x, stats) = fom(A, b, pivoting=pivoting)
    @test stats.solved

    # Left preconditioning
    A = ones(10,10) + 9 * I
    b = 10 * [1:10;]
    M = 1/10 * opEye(10)
    (x, stats) = fom(A, b, M=M, pivoting=pivoting)
    show(stats)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Right preconditioning
    A = ones(10,10) + 9 * I
    b = 10 * [1:10;]
    N = 1/10 * opEye(10)
    (x, stats) = fom(A, b, N=N, pivoting=pivoting)
    show(stats)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)

    # Split preconditioning
    A = ones(10,10) + 9 * I
    b = 10 * [1:10;]
    M = 1/√10 * opEye(10)
    N = 1/√10 * opEye(10)
    (x, stats) = fom(A, b, M=M, N=N, pivoting=pivoting)
    show(stats)
    r = b - A * x
    resid = norm(r) / norm(b)
    @printf("FOM: Relative residual: %8.1e\n", resid)
    @test(resid ≤ fom_tol)
    @test(stats.solved)
  end
end

test_fom()
