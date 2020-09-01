## Matrix-free operators

All methods are matrix free, which means that you only need to provide operator-vector products.

The `A`, `M` or `N` input arguments of Krylov.jl solvers can be any object that represents a linear operator. That object must implement `*`, for multiplication with a vector, `size()` and `eltype()`. For certain methods it must also implement `adjoint()`.

Some methods only require `A * v` products, whereas other ones also require `A' * u` products. In the latter case, `adjoint(A)` must also be implemented.

| A * v                                  | A * v and A' * u                         |
|:--------------------------------------:|:----------------------------------------:|
| CG, CR                                 | CGLS, CRLS, CGNE, CRMR                   |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR   |
| DQGMRES, DIOM                          | BiLQ, QMR, BiLQR, USYMLQ, USYMQR, TriLQR |
| CGS                                    | TriCG, TriMR, USYMLQR                    |

We strongly recommend [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) to model matrix-free operators, but other packages such as [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl), [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) or your own operator can be used as well.

With `LinearOperators.jl`, operators are defined as

```julia
A = LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
```

where
* `type` is the operator element type;
* `nrow` and `ncol` are its dimensions;
* `symmetric` and `hermitian` should be set to `true` or `false`;
* `prod(v)`, `tprod(w)` and `ctprod(u)` are called when writing `A * v`, `tranpose(A) * w`, and `A' * u`, respectively.

See the [tutorial](https://juliasmoothoptimizers.github.io/JSOTutorials.jl/linear-operators/introduction-to-linear-operators/introduction-to-linear-operators.html) and the detailed [documentation](https://juliasmoothoptimizers.github.io/LinearOperators.jl/latest/) for more informations on `LinearOperators.jl`.

## Examples

In the field of nonlinear optimization, finding critical points of a continuous function frequently involves linear systems with a Hessian or Jacobian as coefficient. Materializing such operators as matrices is expensive in terms of operations and memory consumption and is unreasonable for high-dimensional problems. However, it is often possible to implement efficient Hessian-vector and Jacobian-vector products, for example with the help of automatic differentiation tools, and used within Krylov solvers. We now illustrate variants with explicit matrices and with matrix-free operators for two well-known optimization methods.

### Example 1: Newton's Method for convex optimization

At each iteration of Newton's method applied to a $\mathcal{C}^2$ strictly convex function $f : \mathbb{R}^n \rightarrow \mathbb{R}$, a descent direction direction is determined by minimizing the quadratic Taylor model of $f$:

```math
\min_{d \in \mathbb{R}^n}~~f(x_k) + \nabla f(x_k)^T d + \tfrac{1}{2}~d^T \nabla^2 f(x_k) d
```
which is equivalent to solving the symmetric and positive-definite system
```math
\nabla^2 f(x_k) d  = -\nabla f(x_k).
```
The system above can be solved with the conjugate gradient method as follows, using the explicit Hessian:
```@nlp
using ForwardDiff, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(x) = ForwardDiff.hessian(f, x)

d, stats = cg(H(xk), -g(xk))
```

The explicit Hessian can be replaced by a linear operator that only computes Hessian-vector products:

```@example hessian_operator
using ForwardDiff, LinearOperators, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(v) = ForwardDiff.derivative(t -> g(xk + t * v), 0.0)
opH = LinearOperator(Float64, 4, 4, true, true, v -> H(v))

cg(opH, -g(xk))
```

### Example 2: The Gauss-Newton Method for Nonlinear Least Squares

At each iteration of the Gauss-Newton method applied to a nonlinear least-squares objective $f(x) = \tfrac{1}{2}\| F(x)\|^2$ where $F : \mathbb{R}^n \rightarrow \mathbb{R}^m$ is $\mathcal{C}^1$, we solve the subproblem:

```math
\min_{d \in \mathbb{R}^n}~~\tfrac{1}{2}~\|J(x_k) d + F(x_k)\|^2,
```
where $J(x)$ is the Jacobian of $F$ at $x$.

An appropriate iterative method to solve the above linear least-squares problems is LSMR. We could pass the explicit Jacobian to LSMR as follows:
```@nls
using ForwardDiff, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(x) = ForwardDiff.jacobian(F, x)

d, stats = lsmr(J(xk), -F(xk))
```

However, the explicit Jacobian can be replaced by a linear operator that only computes Jacobian-vector and transposed Jacobian-vector products:
```@example jacobian_operator
using LinearAlgebra, ForwardDiff, LinearOperators, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(v) = ForwardDiff.derivative(t -> F(xk + t * v), 0)
Jᵀ(u) = ForwardDiff.gradient(x -> dot(F(x), u), xk)
opJ = LinearOperator(Float64, 3, 2, false, false, v -> J(v), w -> Jᵀ(w), u -> Jᵀ(u))

lsmr(opJ, -F(xk))
```

Note that preconditioners can be also implemented as abstract operators.
For instance, we could compute the Cholesky factorization of $M$ and $N$ and create linear operators that perform the forward and backsolves.

Krylov methods combined with matrix free operators allow to reduce computation time and memory requirements considerably by avoiding building and storing the system matrix.
In the field of partial differential equations, the implementation of high-performance matrix free operators and assembly free preconditioning is a subject of active research.
