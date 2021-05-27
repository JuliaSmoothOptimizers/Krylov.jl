include("get_div_grad.jl")
include("gen_lsq.jl")
include("check_min_norm.jl")

# Symmetric and positive definite systems.
function symmetric_definite(n :: Int=10)
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  b = A * [1:n;]
  return A, b
end

# Symmetric and indefinite systems.
function symmetric_indefinite(n :: Int=10)
  A = spdiagm(-1 => ones(n-1), 0 => 4*ones(n), 1 => ones(n-1))
  A = A - 3 * I
  b = A * [1:n;]
  return A, b
end

# Nonsymmetric and positive definite systems.
function nonsymmetric_definite(n :: Int=10)
  A = [i == j ? 1.0*n : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  return A, b
end

# Nonsymmetric and indefinite systems.
function nonsymmetric_indefinite(n :: Int=10)
  A = [i == j ? n*(-1.0)^(i*j) : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  return A, b
end

# Underdetermined and consistent systems.
function under_consistent(n :: Int=10, m :: Int=25)
  n < m || error("Square or overdetermined system!")
  A = [i/j - j/i for i=1:n, j=1:m]
  b = A * ones(m)
  return A, b
end

# Underdetermined and inconsistent systems.
function under_inconsistent(n :: Int=10, m :: Int=25)
  n < m || error("Square or overdetermined system!")
  A = ones(n, m)
  b = [i == 1 ? -1.0 : 1.0*i for i=1:n]
  return A, b
end

# Square and consistent systems.
function square_consistent(n :: Int=10)
  A = [i/j - j/i for i=1:n, j=1:n]
  b = A * ones(n)
  return A, b
end

# Square and inconsistent systems.
function square_inconsistent(n :: Int=10)
  A = eye(n); A[1, 1] = 0.0
  b = ones(n)
  return A, b
end

# Overdetermined and consistent systems.
function over_consistent(n :: Int=25, m :: Int=10)
  n > m || error("Underdetermined or square system!")
  A = [i/j - j/i for i=1:n, j=1:m]
  b = A * ones(m)
  return A, b
end

# Overdetermined and inconsistent systems.
function over_inconsistent(n :: Int=25, m :: Int=10)
  n > m || error("Underdetermined or square system!")
  A = ones(n, m)
  b = [i == 1 ? -1.0 : 1.0*i for i=1:n]
  return A, b
end

# Sparse Laplacian.
function sparse_laplacian(n :: Int=16)
  A = get_div_grad(n, n, n)
  b = ones(n^3)
  return A, b
end

# Large-scale unsymmetric systems generated with Kronecker products.
function kron_unsymmetric(n :: Int=64)
  N = n^3
  A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(3.0, n), 1 => fill(-2.0, n - 1))
  Id = eye(n)
  A = kron(A, Id) + kron(Id, A)
  A = kron(A, Id) + kron(Id, A)
  x = ones(N)
  b = A * x
  return A, b
end

# Symmetric, indefinite and almost singular systems.
function almost_singular(n :: Int=16)
  A = get_div_grad(n, n, n)
  A = A - 5 * I
  b = A * ones(n^3)
  return A, b
end

# Symmetric, singular and consistent systems.
function singular_consistent(n :: Int=10)
  A = [1.0*i*j for i=1:n, j=1:n] + 5 * eye(n)
  A[:,1] .= A[:,2] .= A[2,:] .= A[1,:] .= 1.0
  b = A * ones(n)
  return A, b
end

# System that cause a breakdown with the symmetric Lanczos process.
function symmetric_breakdown()
  A = [0.0 1.0; 1.0 0.0]
  b = [1.0; 0.0]
  return A, b
end

# System that cause a breakdown with the Lanczos biorthogonalization
# and the orthogonal tridiagonalization processes.
function unsymmetric_breakdown()
  A = [0.0 1.0; -1.0 0.0]
  b = [1.0; 0.0]
  c = [-1.0; 0.0]
  return A, b, c
end

# Underdetermined consistent adjoint systems.
function underdetermined_adjoint(n :: Int=100, m :: Int=200)
  n < m || error("Square or overdetermined system!")
  A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:m]
  b = A * [1:m;]
  c = A' * [-n:-1;]
  return A, b, c
end

# Square consistent adjoint systems.
function square_adjoint(n :: Int=100)
  A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:n]
  b = A * [1:n;]
  c = A' * [-n:-1;]
  return A, b, c
end

# Adjoint systems with Ax = b underdetermined consistent and Aᵀt = c overdetermined insconsistent.
function rectangular_adjoint(n :: Int=10, m :: Int=25)
  Aᵀ, c = over_inconsistent(m, n)
  A = transpose(Aᵀ)
  b = A * ones(m)
  return A, b, c
end

# Overdetermined consistent adjoint systems.
function overdetermined_adjoint(n :: Int=200, m :: Int=100)
  n > m || error("Underdetermined or square system!")
  A = [i == j ? 10.0 : i < j ? 1.0 : -1.0 for i=1:n, j=1:m]
  b = A * [1:m;]
  c = A' * [-n:-1;]
  return A, b, c
end

# Adjoint ODEs.
function adjoint_ode(n :: Int=50)
  χ₁ = χ₂ = χ₃ = 1.0
  # Primal ODE
  # χ₁ * d²U(x)/dx² + χ₂ * dU(x)/dx + χ₃ * U(x) = f(x)
  # U(0) = U(1) = 0
  function f(x)
    return (- χ₁ * π * π + χ₃) .* sin.(π.*x) .+ (χ₂ * π) .* cos.(π.*x)
  end
  # Dual ODE
  # χ₁ * d²V(x)/dx² - χ₂ * dV(x)/dx + χ₃ *  V(x) = g(x)
  # V(0) = V(1) = 0
  function g(x)
    exp.(x)
  end
  A, b, c = ODE(n, f, g, [χ₁, χ₂, χ₃])
  return A, b, c
end

# Adjoint PDEs.
function adjoint_pde(n :: Int=50, m :: Int=50)
  κ₁ = 5.0
  κ₂ = 20.0
  κ₃ = 0.0
  # Primal PDE
  # κ₁ * ( ∂²u(x,y)/∂x² + ∂²u(x,y)/∂y² )  + κ₂ * ( ∂u(x,y)/∂x + ∂u(x,y)/∂y ) = f(x,y), (x,y) ∈ Ω
  # u(x,y) = 0, (x,y) ∈ ∂Ω
  function f(x, y)
    return (-2 * κ₁ * π * π + κ₃) * sin(π * x) * sin(π * y) + κ₂ * π * cos(π * x) * sin(π * y) + κ₂ * π * sin(π * x) * cos(π * y)
  end
  # Dual PDE
  # κ₁ * ( ∂²v(x,y)/∂x² + ∂²v(x,y)/∂y² )  - κ₂ * ( ∂v(x,y)/∂x + ∂v(x,y)/∂y ) = g(x,y), (x,y) ∈ Ω
  # v(x,y) = 0, (x,y) ∈ ∂Ω
  function g(x, y)
    return exp(x + y)
  end
  A, b, c = PDE(n, m, f, g, [κ₁, κ₁, κ₂, κ₂, κ₃])
  return A, b, c
end

# Poisson equation in polar coordinates with homogeneous boundary conditions.
function polar_poisson(n :: Int=50, m :: Int=50)
  f(r, θ) = -3.0 * cos(θ)
  g(r, θ) = 0.0
  A, b = polar_poisson(n, m, f, g)
  return A, b
end

# Poisson equation in cartesian coordinates with homogeneous boundary conditions.
function cartesian_poisson(n :: Int=50, m :: Int=50)
  f(x, y) = - 2.0 * π * π * sin(π * x) * sin(π * y)
  g(x, y) = 0.0
  A, b = cartesian_poisson(n, m, f, g)
  return A, b
end

# Square and preconditioned problems.
function square_preconditioned(n :: Int=10)
  A   = ones(n, n) + (n-1) * eye(n)
  b   = 10.0 * [1:n;]
  M⁻¹ = 1/n * eye(n)
  return A, b, M⁻¹
end

# Square problems with two preconditioners.
function two_preconditioners(n :: Int=10, m :: Int=20)
  A   = ones(n, n) + (n-1) * eye(n)
  b   = ones(n)
  M⁻¹ = 1/√n * eye(n)
  N⁻¹ = 1/√m * eye(n)
  return A, b, M⁻¹, N⁻¹
end

# Random Ax = b with b == 0.
function zero_rhs(n :: Int=10)
  A = rand(n, n)
  b = zeros(n)
  return A, b
end

# Regularized problems.
function regularization(n :: Int=5)
  A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
  b = ones(n)
  λ = 4.0
  return A, b, λ
end

# Saddle-point systems with square A.
function saddle_point(n :: Int=5)
  A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
  b = ones(n)
  D = diagm(0 => [2.0 * i for i = 1:n])
  return A, b, D
end

# Saddle-point systems with rectangular A.
function small_sp(transpose :: Bool=false)
  A = [1.0 0.0; 0.0 -1.0; 3.0 0.0]
  A = transpose ? Matrix(A') : A
  n, m = size(A)
  b = ones(n)
  c = -ones(m)
  D = diagm(0 => [2.0 * i for i = 1:n])
  return A, b, c, D
end

# Symmetric and quasi-definite systems with square A.
function sqd(n :: Int=5)
  A = [2^(i/j)*j + (-1)^(i-j) * n*(i-1) for i = 1:n, j = 1:n]
  b = ones(n)
  M = diagm(0 => [3.0 * i for i = 1:n])
  N = diagm(0 => [5.0 * i for i = 1:n])
  return A, b, M, N
end

# Symmetric and quasi-definite systems with rectangular A.
function small_sqd(transpose :: Bool=false)
  A = [1.0 0.0; 0.0 -1.0; 3.0 0.0]
  A = transpose ? Matrix(A') : A
  n, m = size(A)
  b = ones(n)
  c = -ones(m)
  M = diagm(0 => [3.0 * i for i = 1:n])
  N = diagm(0 => [5.0 * i for i = 1:m])
  return A, b, c, M, N
end
