## Custom workspaces for the Poisson equation with halo regions

### Introduction

The Poisson equation is a fundamental partial differential equation (PDE) in physics and mathematics, modeling phenomena like temperature distribution and incompressible fluid flow.
In a 2D Cartesian domain, it can be expressed as:

```math
\nabla^2 u(x, y) = f(x, y)
```

Here, $u(x, y)$ is the potential function and $f(x, y)$ represents the source term within the domain.

This page explains how to use a Krylov method to solve the Poisson equation over a rectangular region with specified boundary conditions, detailing the use of a Laplacian operator within a data structure that incorporates **halo regions**.

### Finite difference discretization

We solve the Poisson equation numerically by discretizing the 2D domain using a finite difference method.
For a square domain $[0, L] \times [0, L]$, divided into a grid of points, each point approximates the solution $u$ at that position.

With grid spacings $h_x = \frac{L}{N_x + 1}$ and $h_y = \frac{L}{N_y + 1}$, let $u_{i,j}$ denote the approximation of $u(x_i, y_j)$ at grid point $(x_i, y_j) = (ih, jh)$.
The 2D Laplacian can be approximated at each interior grid point $(i, j)$ by combining the following central difference formulas:

```math
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}
```

```math
\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}
```

This yields the discrete Poisson equation:

```math
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = f_{i,j}
```

resulting in a system of linear equations for the $N^2$ unknowns $u_{i,j}$ at each interior grid point.

### Boundary conditions

Boundary conditions complete the system. Common choices are:

- **Dirichlet**: Specifies values of $u$ on the boundary.
- **Neumann**: Specifies the normal derivative (or flux) of $u$ on the boundary.

### Implementing halo regions with HaloVector

In parallel computing, **halo regions** (or ghost cells) around the grid store boundary values from neighboring subdomains, allowing independent stencil computation near boundaries.
This setup streamlines boundary management in distributed environments.

For specialized applications, Krylov.jl’s internal storage expects an `AbstractVector`, which can benefit from a structured data layout.
A **`HaloVector`** provides this structure, using halo regions to enable finite difference stencils without boundary condition checks.
The `OffsetArray` type from [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl) facilitates custom indexing, making it ideal for grids with halo regions.
By embedding an `OffsetArray` within `HaloVector`, we achieve seamless grid alignment, allowing **"if-less"** stencil application.

This setup reduces boundary condition checks in the core loop, yielding clearer and faster code.
The flexible design of `HaloVector` supports 1D, 2D, or 3D configurations, adapting easily to different grid layouts.

### Definition and usage of the HaloVector

`HaloVector` is a specialized vector for grid-based computations, especially finite difference methods with halo regions.
It is parameterized by:

- **`FC`**: The element type of the vector.
- **`D`**: The data array type, which uses `OffsetArray` to enable custom indexing.

```julia
using OffsetArrays

struct HaloVector{FC, D} <: AbstractVector{FC}
    data::D

    function HaloVector(data::D) where {D}
        FC = eltype(data)
        return new{FC, D}(data)
    end
end

function HaloVector{FC,D}(::UndefInitializer, l::Int64) where {FC,D}
    m = n = sqrt(l) |> Int
    data = zeros(FC, m + 2, n + 2)
    v = OffsetMatrix(data, 0:m + 1, 0:n + 1)
    return HaloVector(v)
end

function Base.length(v::HaloVector)
    m, n = size(v.data)
    l = (m - 2) * (n - 2)
    return l
end

function Base.size(v::HaloVector)
    l = length(v)
    return (l,)
end

function Base.getindex(v::HaloVector, idx)
    m, n = size(v.data)
    row = div(idx - 1, n - 2) + 1
    col = mod(idx - 1, n - 2) + 1
    return v.data[row, col]
end
```

The `size` and `getindex` functions support REPL display, aiding interaction, though they are optional for Krylov.jl’s functionality.

### Efficient stencil implementation

Using `HaloVector` with `OffsetArray`, we can apply the discrete Laplacian operator in a matrix-free approach with a 5-point stencil, managing halo regions effectively.
This layout allows **clean and efficient Laplacian computation** without boundary checks within the core loop.

```julia
# Define a matrix-free Laplacian operator
struct LaplacianOperator
    Nx::Int        # Number of grid points in the x-direction
    Ny::Int        # Number of grid points in the y-direction
    Δx::Float64    # Grid spacing in the x-direction
    Δy::Float64    # Grid spacing in the y-direction
end

# Define size and element type for the operator
Base.size(A::LaplacianOperator) = (A.Nx * A.Ny, A.Nx * A.Ny)
Base.eltype(A::LaplacianOperator) = Float64

function LinearAlgebra.mul!(y::HaloVector{Float64}, A::LaplacianOperator, u::HaloVector{Float64})
    # Apply the discrete Laplacian in 2D
    for i in 1:A.Nx
        for j in 1:A.Ny
            # Calculate second derivatives using finite differences
            dx2 = (u.data[i-1,j] - 2 * u.data[i,j] + u.data[i+1,j]) / (A.Δx)^2
            dy2 = (u.data[i,j-1] - 2 * u.data[i,j] + u.data[i,j+1]) / (A.Δy)^2
            
            # Update the output vector with the Laplacian result
            y.data[i,j] = dx2 + dy2
        end
    end

    return y
end
```

### Methods to overload for compatibility with Krylov.jl

To integrate `HaloVector` with Krylov.jl, we define essential vector operations, including dot products, norms, scalar multiplication, and element-wise updates.
These implementations allow Krylov.jl to leverage custom vector types, enhancing both solver flexibility and performance.

```julia
using Krylov
import Krylov.FloatOrComplex

function Krylov.kdot(n::Integer, x::HaloVector{T}, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    res = zero(T)
    for i = 1:mx-1
        for j = 1:nx-1
            res += _x[i,j] * _y[i,j]
        end
    end
    return res
end

function Krylov.knorm(n::Integer, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    res = zero(T)
    for i = 1:mx-1
        for j = 1:nx-1
            res += _x[i,j]^2
        end
    end
    return sqrt(res)
end

function Krylov.kscal!(n::Integer, s::T, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for i = 1:mx-1
        for j = 1:nx-1
            _x[i,j] = s * _x[i,j]
        end
    end
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::HaloVector{T}, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i = 1:mx-1
        for j = 1:nx-1
            _y[i,j] += s * _x[i,j]
        end
    end
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::HaloVector{T}, t::T, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i = 1:mx-1
        for j = 1:nx-1
            _y[i,j] = s * _x[i,j] + t * _y[i,j]
        end
    end
    return y
end

function Krylov.kcopy!(n::Integer, y::HaloVector{T}, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i = 1:mx-1
        for j = 1:nx-1
            _y[i,j] = _x[i,j]
        end
    end
    return y
end

function Krylov.kfill!(x::HaloVector{T}, val::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for i = 1:mx-1
        for j = 1:nx-1
            _x[i,j] = val
        end
    end
    return x
end

function Krylov.kref!(n::Integer, x::HaloVector{T}, y::HaloVector{T}, c::T, s::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i = 1:mx-1
        for j = 1:nx-1
            x_ij = _x[i,j]
            y_ij = _y[i,j]
            _x[i,j] = c       * x_ij + s * y_ij
            _x[i,j] = conj(s) * x_ij - c * y_ij
        end
    end
    return x, y
end
```

Note that `Krylov.kref!` is only required for `minres_qlp`.

### 2D Poisson equation solver with Krylov methods

```julia
using Krylov, LinearAlgebra, OffsetArrays

# Parameters
L = 1.0            # Length of the square domain
Nx = 200           # Number of interior grid points in x
Ny = 200           # Number of interior grid points in y
Δx = L / (Nx + 1)  # Grid spacing in x
Δy = L / (Ny + 1)  # Grid spacing in y

# Define the source term f(x,y)
f(x,y) = -2 * π * π * sin(π * x) * sin(π * y)

# Create the matrix-free Laplacian operator
A = LaplacianOperator(Nx, Ny, Δx, Δy)

# Create the right-hand side
rhs = zeros(Float64, Nx+2, Ny+2)
data = OffsetArray(rhs, 0:Nx+1, 0:Ny+1)
for i in 1:Nx
    for j in 1:Ny
        xi = i * Δx
        yj = j * Δy
        data[i,j] = f(xi, yj)
    end
end
b = HaloVector(data)

# Solve the system with CG
u_sol, stats = Krylov.cg(A, b, atol=1e-12, rtol=0.0, verbose=1)

# The exact solution is u(x,y) = sin(πx) * sin(πy)
u_star = [sin(π * i * Δx) * sin(π * j * Δy) for i=1:Nx, j=1:Ny]
norm(u_sol.data[1:Nx, 1:Ny] - u_star, Inf)
```

### Conclusion

Implementing a 2D Poisson equation solver with `HaloVector` improves code clarity and efficiency.
Custom indexing with `OffsetArray` streamlines halo region management, eliminating boundary checks within the core loop.
This approach reduces branching, yielding faster execution, especially on large grids.
`HaloVector`'s flexibility also makes it easy to extend to 3D grids or more complex stencils.

!!! info
    [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) uses a similar strategy with its `Field` type, efficiently solving large linear systems with Krylov.jl.
