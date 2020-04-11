## Matrix-free operators

All methods are matrix-free which means that you only need to provide matrix-vector products.

For `A`, `M` or `N` parameters used in Krylov.jl solvers you can pass any linear operator that materialize the matrix. Thereafter, the solver will use `op * v` for operator-vector products. `size(op)` and `eltype(op)` must be implemented.

Some methods only required $A * v$ products, whereas other ones also required $A^T * u$ products. In that case, `adjoint(A)` must be implemented too.

| A * v                                  | A * v and Aᵀ * u                       |
|:--------------------------------------:|:--------------------------------------:|
| CG, CR                                 | CGLS, CRLS, CGNE, CRMR                 |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR |
| DQGMRES, DIOM                          | BiLQ, QMR, BiLQR                       |
| CGS                                    | USYMLQ, USYMQR, TriLQR, USYMLQR        |

## Tutorial

We strongly recommend the package [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) if you want to modelize matrix-free operators but other packages like [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl), [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) or your own operator can be used too.

```julia
A = LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
```

`type` is the floating-point system in which matrix-vector products are computed (Float32, Float64, BigFloat, ...). `nrows`  and `ncols` are the dimension of the operators. Properties of the linear operator are given by `symmetric` and `hermitian` booleans. `prod(v)`, `tprod(u)` and `ctprod(w)` are the functions used when `A * v`, `transpose(A) * u` and `adjoint(A) * w` are respectively called.

Detailed documentation about `LinearOperators.jl` is available [here](https://juliasmoothoptimizers.github.io/LinearOperators.jl/latest/).

Given a convex quadratic function $f(x) = (x_1 - 1)^2 + (x_2 - 2)^2 + (x_3 - 3)^2$, you should be interested by determining the minimum $d$ of this function by solving $\nabla^2 f(x_0) d = - \nabla f(x_0)$ because
```math
f(d) = d^T \nabla^2 f(x_0) d + \nabla f(x_0)^T d + f(x_0).
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

The main problem is that $\nabla^2(x_0)$ is stored as a dense matrix... We can avoid that by using a linear operator and not forming explicitly the hessian!

```@example linear_operator
using ForwardDiff, LinearOperators, Krylov

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2
g(x) = ForwardDiff.gradient(f, x)

x0 = zeros(3)
H(v) = ForwardDiff.derivative(t -> g(x0 + t * v), 0.0)
opH = LinearOperator(Float64, 3, 3, true, true, v -> H(v))
cg(opH, -g(x0))
```
