using Krylov
using LinearAlgebra, Printf

A = diagm([1.0; 2.0; 3.0; 0.0])
n = size(A, 1)
b = [1.0; 2.0; 3.0; 4.0]
b_norm = norm(b)

# MINRES-QLP returns the minimum-norm solution of symmetric, singular and inconsistent systems
(x, stats) = minres_qlp(A, b);
r = b - A * x;

@printf("Residual r: %s\n", Krylov.vec2str(r))
@printf("Relative residual norm ‖r‖: %8.1e\n", norm(r) / b_norm)
@printf("Solution x: %s\n", Krylov.vec2str(x))
@printf("Minimum-norm solution? %s\n", x ≈ [1.0; 1.0; 1.0; 0.0])
