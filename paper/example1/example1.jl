using LinearAlgebra    # Linear algebra library of Julia
using SparseArrays     # Sparse library of Julia
using Krylov           # Krylov methods and processes
using LinearOperators  # Linear operators
using ForwardDiff      # Automatic differentiation
using Quadmath         # Quadruple precision
using MKL              # Intel BLAS

"The Gauss-Newton method for Nonlinear Least Squares"
function gauss_newton(F, JF, x₀::AbstractVector{T}; itmax = 200, tol = √eps(T)) where T
    n = length(x₀)
    x = copy(x₀)
    Fx = F(x)
    m = length(Fx)
    iter = 0
    S = typeof(x)                 # precision and architecture
    solver = LsmrSolver(m, n, S)  # structure that contains the workspace of LSMR
    solved = tired = false
    while !(solved || tired)
        Jx = JF(x)              # Compute J(xₖ)
        lsmr!(solver, Jx, -Fx)  # Minimize ‖J(xₖ)Δx + F(xₖ)‖
        x .+= solver.x          # Update xₖ₊₁ = xₖ + Δx
        Fx_old = Fx             # F(xₖ)
        Fx = F(x)               # F(xₖ₊₁)
        iter += 1
        solved = norm(Fx - Fx_old) / norm(Fx) ≤ tol
        tired = iter ≥ itmax
    end
    return x
end

T = Float128  # IEEE quadruple precision
x₀ = ones(T, 2)
F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]         # F(x)
J(y, x, v) = ForwardDiff.derivative!(y, t -> F(x + t * v), 0)  # y ← JF(x)v
Jᵀ(y, x, w) = ForwardDiff.gradient!(y, x -> dot(F(x), w), x)   # y ← JFᵀ(x)w
symmetric = hermitian = false
JF(x) = LinearOperator(T, 3, 2, symmetric, hermitian, (y, v) -> J(y, x, v),   # non-transpose
                                                      (y, w) -> Jᵀ(y, x, w),  # transpose
                                                      (y, w) -> Jᵀ(y, x, w))  # conjugate transpose
xstar = gauss_newton(F, JF, x₀)
