using LinearAlgebra    # Linear algebra library of Julia
using SparseArrays     # Sparse library of Julia
using Test             # Test library of Julia
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
x_exact = T[8, 0.25]
x₀ = ones(T, 2)
t = T[1, 2, 3, 4, 5, 6, 7, 8]
y = [trunc(x_exact[1] * exp(x_exact[2] * t[i]), digits=3) for i=1:8]
F(x) = [x[1] * exp(x[2] * t[i]) - y[i] for i=1:8]              # F(x)
J(y, x, v) = ForwardDiff.derivative!(y, h -> F(x + h * v), 0)  # y ← JF(x)v
Jᵀ(y, x, w) = ForwardDiff.gradient!(y, x -> dot(F(x), w), x)   # y ← JFᵀ(x)w
symmetric = hermitian = false
JF(x) = LinearOperator(T, 8, 2, symmetric, hermitian, (y, v) -> J(y, x, v),   # non-transpose
                                                      (y, w) -> Jᵀ(y, x, w),  # transpose
                                                      (y, w) -> Jᵀ(y, x, w))  # conjugate transpose
x = gauss_newton(F, JF, x₀)

# Check the solution returned by the Gauss-Newton method
@test norm(x - x_exact) ≤ 1e-4
