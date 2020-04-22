## Matrix-free operators

All methods are matrix-free, which means that you only need to provide operator-vector products.

The `A`, `M` or `N` input arguments of Krylov.jl solvers can be any object that represents a linear operator. That object must implement `*`, for multiplication with a vector, `size()` and `eltype()`. For certain methods it must also implement `adjoint()`.

Some methods only require `A * v` products, whereas other ones also require `A' * u` products. In the latter case, `adjoint(A)` must also be implemented.

| A * v                                  | A * v and A' * u                       |
|:--------------------------------------:|:--------------------------------------:|
| CG, CR                                 | CGLS, CRLS, CGNE, CRMR                 |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR |
| DQGMRES, DIOM                          | BiLQ, QMR, BiLQR                       |
| CGS                                    | USYMLQ, USYMQR, TriLQR, USYMLQR        |

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

In the field of non-linear optimization, find critical points of a continuous function frequently involve linear systems with hessian and jacobian matrices. Form explicitly these matrices is expensive in term of operations and memory and is unreasonable for high dimensional problems. However efficient hessian-vector and jacobian-vector products can be computed with automatic differentiation tools and used within Krylov solvers. Variants without and with matrix-free operators are presented for two well-known optimization methods.

At each iteration of the **Newton** method applied on a convex function $f : \mathbb{R}^n \rightarrow \mathbb{R}$, a descent direction direction is determined by minimizing the quadratic Taylor model of $f$ :

```math
\min_{d \in \mathbb{R}^n}~~\tfrac{1}{2}~d^T \nabla^2 f(x_k) d + \nabla f(x_k)^T d + f(x_k) \\
\Leftrightarrow \\
\nabla^2 f(x_k) d  = -\nabla f(x_k).
```

```@nlp
using ForwardDiff, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(x) = ForwardDiff.hessian(f, x)

d, stats = cg(H(xk), -g(xk))
```

The hessian matrix can be replaced by a linear operator that only computes hessian-vector products.

```@example hessian_operator
using ForwardDiff, LinearOperators, Krylov

xk = -ones(4)

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2 + (x[4] - 4)^2

g(x) = ForwardDiff.gradient(f, x)

H(v) = ForwardDiff.derivative(t -> g(xk + t * v), 0.0)
opH = LinearOperator(Float64, 4, 4, true, true, v -> H(v))

cg(opH, -g(xk))
```

At each iteration of the **Gauss-Newton** method applied on a non-linear least squares problem $f(x) = \tfrac{1}{2}\| F(x)\|^2$ where $F : \mathbb{R}^n \rightarrow \mathbb{R}^m$, the following subproblem needs to be solved :

```math
\min_{d \in \mathbb{R}^n}~~\tfrac{1}{2}~\|J(x_k) d + F(x_k)\|^2 \\
\Leftrightarrow \\
J(x_k)^T J(x_k) d  = J(x_k)^T F(x_k).
```

```@nls
using ForwardDiff, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(x) = ForwardDiff.jacobian(F, x)

d, stats = lsmr(J(xk), -F(xk))
```

The jacobian matrix can be replaced by a linear operator that only computes jacobian-vector and transposed jacobian-vector products.

```@example jacobian_operator
using LinearAlgebra, ForwardDiff, LinearOperators, Krylov

xk = ones(2)

F(x) = [x[1]^4 - 3; exp(x[2]) - 2; log(x[1]) - x[2]^2]

J(v) = ForwardDiff.derivative(t -> F(xk + t * v), 0)
Jᵀ(u) = ForwardDiff.gradient(x -> dot(F(x), u), xk)
opJ = LinearOperator(Float64, 3, 2, false, false, v -> J(v), w -> Jᵀ(w), u -> Jᵀ(u))

lsmr(opJ, -F(xk))
```
