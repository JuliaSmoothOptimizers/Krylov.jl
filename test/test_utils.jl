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

# Underdetermined and integer systems.
function under_int(n :: Int=3, m :: Int=5)
  n < m || error("Square or overdetermined system!")
  A = [i^j for i=1:n, j=1:m]
  b = A * ones(Int, m)
  return A, b
end

# Square and integer systems.
function square_int(n :: Int=10)
  A = spdiagm(-1 => ones(Int, n-1), 0 => 4*ones(Int, n), 1 => ones(Int, n-1))
  b = A * [1:n;]
  return A, b
end

# Overdetermined and integer systems.
function over_int(n :: Int=5, m :: Int=3)
  n > m || error("Underdetermined or square system!")
  A = [i^j for i=1:n, j=1:m]
  b = A * ones(Int, m)
  return A, b
end

# Sparse Laplacian.
function sparse_laplacian(n :: Int=16)
  A = get_div_grad(n, n, n)
  b = ones(size(A, 1))
  return A, b
end

# Symmetric, indefinite and almost singular systems.
function almost_singular(n :: Int=16)
  A = get_div_grad(n, n, n)
  A = A - 5 * I
  b = randn(size(A, 1))
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

# Square and preconditioned problems.
function square_preconditioned(n :: Int=10)
  A = ones(n, n) + (n-1) * I
  b = n * [1:n;]
  O = zeros(n, n)
  M = PreallocatedLinearOperator(1/n * (O + I))
  return A, b, M
end

# Square problems with two preconditioners.
function two_preconditioners(n :: Int=10, m :: Int=20)
  A = ones(n, n) + (n-1) * I
  b = ones(n)
  O = zeros(n, n)
  M = PreallocatedLinearOperator(1/√n * (O + I))
  N = PreallocatedLinearOperator(1/√m * (O + I))
  return A, b, M, N
end

# Random Ax = b with b == 0.
function zero_rhs(n :: Int=10)
  A = rand(n, n)
  b = zeros(n)
  return A, b
end
