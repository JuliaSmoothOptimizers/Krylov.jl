# Custom vector type for the 2D Poisson equation with halo regions

## Introduction

The 2D Poisson equation is a fundamental partial differential equation (PDE) widely used in physics and mathematics to model various phenomena, including temperature distribution and incompressible fluid flow.
It can be expressed as:

```math
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)
```

where:
- $u(x, y)$ is the unknown function we seek.
- $f(x, y)$ is a known function (the "source term") representing the distribution of sources or sinks in the domain.

This equation is typically solved over a rectangular domain with boundary conditions specified at the edges.

## Finite difference discretization

To numerically solve the Poisson equation, we employ the finite difference method to approximate the second derivatives on a grid.
This approach involves dividing the domain into a grid of points and using differences between neighboring values to approximate derivatives.

Assuming a square domain $[0, L] \times [0, L]$ discretized with $(N_x+2, N_y+2)$ points along each dimension (with grid spacing $h_x = \frac{L}{N_x+1}$ and $h_y = \frac{L}{N_y+1}$, let $u_{i,j}$ denote the approximation of $u(x_i, y_j)$ at the grid point $(x_i, y_j) = (ih, jh)$.

### Discretized Laplacian

The 2D Laplacian, $\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$, can be approximated at each of the $N^2$ interior grid point $(i, j)$ using the central difference formula:

```math
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}
```

```math
\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}
```

Combining these yields the discrete form of the Poisson equation:

```math
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = f_{i,j}
```

This represents a system of linear equations for the unknowns $u_{i,j}$ at each interior grid point.

### Boundary conditions

To complete the system, we need boundary conditions for $u$ along the domain edges.
Common options include:
- **Dirichlet boundary conditions**: specifying the value of $u$ on the boundary.
- **Neumann boundary conditions**: specifying the derivative (flux) of $u$ normal to the boundary.

## Implementing halo regions with `MyVector`

In practical applications, particularly in parallel computing, it is common to introduce **halo regions** (or ghost cells) around the grid.
These additional layers store boundary values from neighboring subdomains, allowing each subdomain to compute stencils near its boundaries independently without immediate communication with neighboring domains.
Halo regions simplify boundary condition management in distributed or multi-threaded environments.

In Krylov.jl, while internal storage for each Krylov method typically expects an `AbstractVector`, specific applications can benefit from structured data layouts.
This is where a specialized vector type called **`MyVector`** comes into play. 

Using `MyVector` with halo regions enables the implementation of finite difference stencils without boundary condition checks, enhancing both readability and performance.
The **`OffsetArray`** type from the [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl) package supports custom indexing, making it ideal for grids with halo regions.
By wrapping an `OffsetArray` within an `MyVector`, we can access elements using custom offsets that align with the grid's physical layout.
This configuration allows for ``if-less'' stencils, avoiding direct boundary condition checks within the core loop, resulting in cleaner and potentially faster code.

The design of `MyVector` can be easily adapted for 1D, 2D, or 3D problems with minimal changes, providing flexibility in handling various grid configurations.

## Definition of the `MyVector`

The `MyVector` type is a specialized vector designed for efficient handling of grid-based computations, particularly in the context of finite difference methods with halo regions.
It is parameterized by:
- **`FC`**: The element type of the vector.
- **`D`**: The type of the data array, which utilizes `OffsetArray` to enable custom indexing.

Here is the definition of the `MyVector`:

```julia
using OffsetArrays

struct MyVector{FC, D} <: AbstractVector{FC}
    data::D

    function MyVector(data::D) where {D}
        FC = eltype(data)
        return new{FC, D}(data)
    end
end

# Constructor
function MyVector{FC,D}(::UndefInitializer, l::Int64) where {FC,D}
    m = n = sqrt(l) |> Int
    data = zeros(FC, m+2, n+2)
    v = OffsetMatrix(data, 0:m+1, 0:n+1)
    return MyVector(v)
end

function Base.length(v::MyVector)
    m, n = size(v.data)
    l = (m - 2) * (n - 2)
    return l
end

function Base.size(v::MyVector)
    l = length(v)
    return (l,)
end

function Base.getindex(v::MyVector, idx)
    m, n = size(v.data)
    row = div(idx-1, n-2) + 1
    col = mod(idx-1, n-2) + 1
    return v.data[row, col]
end
```

The functions `size` and `getindex` are defined to enable display in the REPL.

## Using `MyVector` for the 2D Poisson equation

By utilizing `MyVector`, we can implement the finite difference stencil for the Laplacian operator efficiently, eliminating the need for conditional checks for boundary elements.

### Stencil implementation

Assuming `data` is initialized as an `OffsetArray` with appropriate halo regions, we can define a matrix-free Laplacian operator and apply a typical 5-point stencil operation as follows:

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

function LinearAlgebra.mul!(y::MyVector{Float64}, A::LaplacianOperator, u::MyVector{Float64})
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

### Benefits of Using `MyVector`

Utilizing `MyVector` offers several significant benefits for solving the 2D Poisson equation:

- **Simplified Code**: The custom indexing capabilities of `OffsetArray` allow for the incorporation of boundary data from halo regions. This eliminates the need for boundary checks within the core loop, resulting in clearer and more maintainable code.

- **Performance**: By removing boundary checks, we reduce branching in the code, which enhances computational efficiency, particularly for large grids.
This leads to faster execution times and better overall performance.

- **Flexibility**: `MyVector` can be easily extended to accommodate more complex stencils or additional dimensions (e.g., 3D grids) by simply adjusting the offsets.
This adaptability makes it a powerful tool for various numerical applications.

By leveraging these advantages, we can efficiently solve the 2D Poisson equation while maintaining a clear and concise code structure.

## Required methods for Krylov.jl compatibility

To integrate `MyVector` with Krylov.jl, the following operations must be defined for compatibility:

```julia
using Krylov
import Krylov.FloatOrComplex

function Krylov.kdot(n::Integer, x::MyVector{T}, y::MyVector{T}) where T <: FloatOrComplex
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

function Krylov.knorm(n::Integer, x::MyVector{T}) where T <: FloatOrComplex
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

function Krylov.kscal!(n::Integer, s::T, x::MyVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for i = 1:mx-1
        for j = 1:nx-1
            _x[i,j] = s * _x[i,j]
        end
    end
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::MyVector{T}, y::MyVector{T}) where T <: FloatOrComplex
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

function Krylov.kaxpby!(n::Integer, s::T, x::MyVector{T}, t::T, y::MyVector{T}) where T <: FloatOrComplex
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

function Krylov.kcopy!(n::Integer, y::MyVector{T}, x::MyVector{T}) where T <: FloatOrComplex
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

function Krylov.kfill!(x::MyVector{T}, val::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for i = 1:mx-1
        for j = 1:nx-1
            _x[i,j] = val
        end
    end
    return x
end
```

These methods enable Krylov.jl to use custom vector types, allowing for operations like dot products, norms, scalar multiplication, and element-wise updates, which are essential for Krylov solvers.

## Complete example

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
b = MyVector(data)

# Solve the system with CG
u_sol, stats = Krylov.cg(A, b, atol=1e-12, rtol=0.0, verbose=1)

# The exact solution is u(x,y) = sin(πx) * sin(πy)
u_star = [sin(π * i * Δx) * sin(π * j * Δy) for i=1:Nx, j=1:Ny]
norm(u_sol.data[1:Nx, 1:Ny] - u_star, Inf)
```

## Conclusion

This overview illustrates the effective implementation of a 2D Poisson equation solver utilizing a specialized vector type storage with `MyVector` to accommodate halo regions.
By leveraging custom indexing, we significantly enhance both code readability and performance, enabling a flexible framework that can be adapted to a variety of applications.

!!! info
    The package [Oceanigans.jl](https://github.com/CliMA/Oceananigans.jl) utilizes a similar approach with its type `Field` to solve linear systems involving millions of variables efficiently with Krylov.jl.
