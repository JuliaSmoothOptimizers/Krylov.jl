using LinearOperators

if VERSION < v"0.4-"
  import IterativeSolvers
end
import KrylovMethods
import Krylov

include("../test/get_div_grad.jl");

for N in [32, 64, 128]
  @printf("N = %d\n", N);
  A = get_div_grad(N, N, N);
  n = size(A, 1);
  b = rand(n);

  # Define a linear operator with preallocation.
  Ap = zeros(n);
  op = LinearOperator(n, Float64, p -> A_mul_B!(1.0, A, p, 0.0, Ap));

  # Everybody stops when ‖r‖/‖r0‖ ≤ rtol.
  rtol = 1.0e-6;

  @printf("Krylov:           ")
  (x, stats) = Krylov.cg(op, b, atol=0.0, rtol=rtol, itmax=size(A, 1));
  @time (x, stats) = Krylov.cg(op, b, atol=0.0, rtol=rtol, itmax=size(A, 1));

  @printf("KrylovMethods:    ")
  x,flag,relres,iter,resvec = KrylovMethods.cg(A, b, tol=rtol, maxIter=size(A, 1));
  @time x,flag,relres,iter,resvec = KrylovMethods.cg(A, b, tol=rtol, maxIter=size(A, 1));

  if VERSION < v"0.4-"
    @printf("IterativeSolvers: ")
    (x, hist) = IterativeSolvers.cg(A, b, tol=rtol, maxiter=size(A, 1));
    @time (x, hist) = IterativeSolvers.cg(A, b, tol=rtol, maxiter=size(A, 1));
  end
end
