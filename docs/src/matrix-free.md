## Matrix-free operators

All methods are matrix-free which means that you only need to provide operator-vector products.

The `A`, `M` or `N` input arguments of Krylov.jl solvers can be any object that represents a linear operator. That object must implement `*`, for multiplication with a vector, `size()` and `eltype()`. For certain methods it must also implement `adjoint()`.

Some methods only required `A * v` products, whereas other ones also required `A' * u` products. In that case, `adjoint(A)` must be implemented too.

| A * v                                  | A * v and A' * u                       |
|:--------------------------------------:|:--------------------------------------:|
| CG, CR                                 | CGLS, CRLS, CGNE, CRMR                 |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR |
| DQGMRES, DIOM                          | BiLQ, QMR, BiLQR                       |
| CGS                                    | USYMLQ, USYMQR, TriLQR, USYMLQR        |

We strongly recommend the package [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) if you want to model matrix-free operators but other packages such as [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl), [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) or your own operator can be used too.

```julia
A = LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
```

* `type` is the operator element type;
* `nrow` and `ncol` are its dimensions;
* `symmetric` and `hermitian` should be set to `true` or `false`;
* `prod(v)`, `tprod(w)` and `ctprod(u)` are called when writing `A * v`, `tranpose(A) * w`, and `A' * u`, respectively.

See the [tutorial](https://juliasmoothoptimizers.github.io/JSOTutorials.jl/linear-operators/introduction-to-linear-operators/introduction-to-linear-operators.html) and the detailed [documentation](https://juliasmoothoptimizers.github.io/LinearOperators.jl/latest/) for more informations on `LinearOperators.jl`.

## Example

Given a convex quadratic function $f(x) = (x_1 - 1)^2 + (x_2 - 2)^2 + (x_3 - 3)^2$, you should be interested by determining the minimum $d$ of this function by solving $\nabla^2 f(x_0) d = - \nabla f(x_0)$ because
```math
f(d) = ½~d^T \nabla^2 f(x_0) d + \nabla f(x_0)^T d + f(x_0).
```

!!! info "Remark"

    If $f$ is just convex, it's the quadratic Taylor model that is minimized.

```@example dense_hessian
using ForwardDiff, Krylov

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2
g(x) = ForwardDiff.gradient(f, x)
H(x) = ForwardDiff.hessian(f, x)

x0 = zeros(3)
d, stats = cg(H(x0), -g(x0))
```

However we can avoid the computation of $\nabla^2 f(x_0)$ by using a linear operator and not forming explicitly the hessian!

```@example linear_operator
using ForwardDiff, LinearOperators, Krylov

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2
g(x) = ForwardDiff.gradient(f, x)

x0 = zeros(3)
H(v) = ForwardDiff.derivative(t -> g(x0 + t * v), 0.0)
opH = LinearOperator(Float64, 3, 3, true, true, v -> H(v))
cg(opH, -g(x0))
```
