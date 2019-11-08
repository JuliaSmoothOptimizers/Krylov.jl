using Krylov
using LinearAlgebra, Printf

m = 5
n = 8
λ = 1.0e-3
A = rand(m, n)
b = A * ones(n)
xy_exact = [A  λ*I] \ b # In Julia, this is the min-norm solution!

(x, y, stats) = craig(A, b, λ=λ, atol=0.0, rtol=1.0e-20, verbose=true)
show(stats)

# Check that we have a minimum-norm solution.
# When λ > 0 we solve min ‖(x,s)‖  s.t. Ax + λs = b, and we get s = λy.
@printf("Primal feasibility: %7.1e\n", norm(b - A * x - λ^2 * y) / norm(b))
@printf("Dual   feasibility: %7.1e\n", norm(x - A' * y) / norm(x))
@printf("Error in x: %7.1e\n", norm(x - xy_exact[1:n]) / norm(xy_exact[1:n]))
if λ > 0.0
  @printf("Error in y: %7.1e\n", norm(λ * y - xy_exact[n+1:n+m]) / norm(xy_exact[n+1:n+m]))
end
