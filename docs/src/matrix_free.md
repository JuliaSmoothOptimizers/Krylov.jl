```@raw html
<style>
.content table td {
    border-right-width: 1px;
}
.content table th {
    border-right-width: 1px;
}
.content table td:last-child {
    border-right-width: 0px;
}
.content table th:last-child {
    border-right-width: 0px;
}
html.theme--documenter-dark .content table td {
    border-right-width: 1px;
}
html.theme--documenter-dark .content table th {
    border-right-width: 1px;
}
html.theme--documenter-dark .content table td:last-child {
    border-right-width: 0px;
}
html.theme--documenter-dark .content table th:last-child {
    border-right-width: 0px;
}
</style>
```

## [Matrix-free operators](@id matrix-free)

All methods are matrix-free, which means that you only need to provide operator-vector products.

The `A` or `B` input arguments of Krylov.jl solvers can be any object that represents a linear operator. That object must implement `mul!`, for multiplication with a vector, `size()` and `eltype()`. For certain methods it must also implement `adjoint()`.

Some methods only require `A * v` products, whereas other ones also require `A' * u` products. In the latter case, `adjoint(A)` must also be implemented.

| A * v                                           | A * v and A' * u                         |
|:-----------------------------------------------:|:----------------------------------------:|
| CG, CR, CAR                                     | CGLS, CRLS, CGNE, CRMR                   |
| SYMMLQ, CG-LANCZOS, MINRES, MINRES-QLP, MINARES | LSLQ, LSQR, LSMR, LNLQ, CRAIG, CRAIGMR   |
| DIOM, FOM, DQGMRES, GMRES, FGMRES, BLOCK-GMRES  | BiLQ, QMR, BiLQR, USYMLQ, USYMQR, TriLQR |
| CGS, BICGSTAB                                   | TriCG, TriMR                             |
| CG-LANCZOS-SHIFT                                | CGLS-LANCZOS-SHIFT                       |

!!! info
    GPMR is the only method that requires `A * v` and `B * w` products.

Preconditioners `M`, `N`, `C`, `D`, `E` or `F` can be also linear operators and must implement `mul!` or `ldiv!`.

We strongly recommend [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) to model matrix-free operators, but other packages such as [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), [DiffEqOperators.jl](https://github.com/SciML/DiffEqOperators.jl) or your own operator can be used as well.

With `LinearOperators.jl`, operators are defined as

```julia
A = LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
```

where
* `type` is the operator element type;
* `nrow` and `ncol` are its dimensions;
* `symmetric` and `hermitian` should be set to `true` or `false`;
* `prod(y, v)`, `tprod(y, w)` and `ctprod(u, w)` are called when writing `mul!(y, A, v)`, `mul!(y, transpose(A), w)`, and `mul!(y, A', u)`, respectively.

See the [tutorial](https://jso.dev/tutorials/introduction-to-linear-operators/) and the detailed [documentation](https://jso.dev/LinearOperators.jl/dev/) for more information on `LinearOperators.jl`.

## Examples with automatic differentiation

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

H(y, v) = ForwardDiff.derivative!(y, t -> g(xk + t * v), 0)
opH = LinearOperator(Float64, 4, 4, true, true, (y, v) -> H(y, v))

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

J(y, v) = ForwardDiff.derivative!(y, t -> F(xk + t * v), 0)
Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
opJ = LinearOperator(Float64, 3, 2, false, false, (y, v) -> J(y, v),
                                                  (y, w) -> Jᵀ(y, w),
                                                  (y, u) -> Jᵀ(y, u))

lsmr(opJ, -F(xk))
```

## Example with FFT and IFFT

### Example 3: Solving the Poisson equation with FFT and IFFT

In applications related to partial differential equations (PDEs), linear systems can arise from discretizing differential operators.
Storing such operators as explicit matrices is computationally expensive and unnecessary when matrix-free methods can be used, particularly with structured grids.

The FFT is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, transforming data from the spatial domain to the frequency domain.
In the context of solving PDEs, it simplifies the application of differential operators like the Laplacian by converting derivatives into algebraic operations.

For a function $u(x)$ discretized on a periodic grid with $n$ points, the FFT of $u$ is:

```math
\hat{u}_k = \sum_{j=0}^{n-1} u_j e^{-i k x_j},
```

where $\hat{u}_k$ represents the Fourier coefficients for the frequency $k$, and $u_j$ is the value of $u$ at the grid point $x_j$ defined as $x_j = \frac{2 \pi j}{L}$ with period $L$.
The inverse FFT (IFFT) reconstructs $u$ from its Fourier coefficients:

```math
u_j = \frac{1}{n} \sum_{k=0}^{n-1} \hat{u}_k e^{i k x_j}.
```

In Fourier space, the Laplacian operator $\frac{d^2}{dx^2}$ becomes a simple multiplication by $-k^2$, where $k$ is the wavenumber derived from the grid size.
This transforms the Poisson equation $\frac{d^2 u(x)}{dx^2} = f(x)$ into an algebraic equation in the frequency domain:

```math
-k^2 \hat{u}_k = \hat{f}_k.
```

By solving for $\hat{u}_k$ and applying the IFFT, we can recover the solution $u(x)$ efficiently.

The inverse FFT is used to convert data from the frequency domain back to the spatial domain.
Once the solution in frequency space is obtained by dividing the Fourier coefficients $\hat{f}_k$ by $-k^2$,
the IFFT is applied to transform the result back to the original grid points in the spatial domain.

In some cases, even though the FFT provides an efficient way to apply differential operators (such as the Laplacian)
in the frequency domain, a direct solution may not be feasible due to complex boundary conditions,
variable coefficients, or grid irregularities.
In these situations, the FFT must be coupled with a Krylov method to iteratively solve the problem.

This example consists of solving the 1D Poisson equation on a periodic domain $[0, 4\pi]$:

```math
\frac{d^2 u(x)}{dx^2} = f(x),
```

where $u(x)$ is the unknown solution, and $f(x)$ is the given source term.
We solve this equation using [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) to compute the matrix-free action of the Laplacian within the conjugate gradient solver.

Note that while a direct FFT-based approach can be used here due to the simplicity of the periodic boundary conditions,
this example illustrates how a Krylov method can be employed to solve more challenging problems.

```@example fft_poisson
using FFTW, Krylov, LinearAlgebra

# Define the problem size and domain
n = 32768                         # Number of grid points (2^15)
L = 4π                            # Length of the domain
x = LinRange(0, L, n+1)[1:end-1]  # Periodic grid (excluding the last point)

# Define the source term f(x)
f = sin.(x)

# Define a matrix-free operator using FFT and IFFT
struct FFTPoissonOperator
    n::Int
    L::Float64
    complex::Bool
    k::Vector{Float64}  # Store Fourier wave numbers
end

function FFTPoissonOperator(n::Int, L::Float64, complex::Bool)
    if complex
        k = Vector{Float64}(undef, n)
    else
        k = Vector{Float64}(undef, n÷2 + 1)
    end
    k[1] = 0  # DC component -- f(x) = sin(x) has a mean of 0
    for j in 1:(n÷2)
        k[j+1] = 2 * π * j / L  # Positive wave numbers
    end
    if complex
        for j in 1:(n÷2 - 1)
            k[n-j+1] = -2 * π * j / L  # Negative wave numbers
        end
    end
    return FFTPoissonOperator(n, L, complex, k)
end

Base.size(A::FFTPoissonOperator) = (n, n)

function Base.eltype(A::FFTPoissonOperator)
    type = A.complex ? ComplexF64 : Float64
    return type
end

function LinearAlgebra.mul!(y::Vector, A::FFTPoissonOperator, u::Vector)
    # Transform the input vector `u` to the frequency domain using `fft` or `rfft`.
    # If the operator is complex, use the full FFT; otherwise, use the real FFT.
    if A.complex
        u_hat = fft(u)
    else
        u_hat = rfft(u)
    end

    # In Fourier space, solve the system by multiplying with -k^2 (corresponding to the second derivative).
    # This step applies the Laplacian operator in the frequency domain.
    u_hat .= -u_hat .* (A.k .^ 2)

    # Transform the result back to the spatial domain using `ifft` or `irfft`.
    # If the operator is complex, use the full inverse FFT; otherwise, use the inverse real FFT.
    if A.complex
        y .= ifft(u_hat)
    else
        y .= irfft(u_hat, A.n)
    end

    return y
end


# Create the matrix-free operator for the Poisson equation
complex = false
A = FFTPoissonOperator(n, L, complex)

# Solve the linear system using CG
u_sol, stats = cg(A, f, atol=1e-10, rtol=0.0, verbose=1)

# The exact solution is u(x) = -sin(x)
u_star = -sin.(x)
u_star ≈ u_sol
```

Note that preconditioners can be also implemented as abstract operators.
For instance, we could compute the Cholesky factorization of $M$ and $N$ and create linear operators that perform the forward and backsolves.

Krylov methods combined with factorization free operators allow to reduce computation time and memory requirements considerably by avoiding building and storing the system matrix.
In the field of partial differential equations, the implementation of high-performance factorization free operators and assembly free preconditioning is a subject of active research.
