## Custom workspaces for the Poisson equation with halo regions

### Introduction

The Poisson equation is a fundamental partial differential equation (PDE) commonly used in physics and mathematics to model phenomena such as temperature distribution and incompressible fluid flow.
In a 2D Cartesian domain, it can be expressed as:

```math
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)
```

Here, $u(x, y)$ represents the potential function we seek, while $f(x, y)$ is a known function (the "source term") that indicates the distribution of sources or sinks within the domain.
This equation is typically solved over a rectangular region with boundary conditions specified at the edges.

This overview will discuss the numerical methods used to solve the Poisson equation, emphasizing the application of a Laplacian operator on a specialized data structure that incorporates halo regions. 

### Finite difference discretization

To numerically solve the Poisson equation, we use the finite difference method to approximate the second derivatives on a grid.
This approach involves dividing the domain into a grid of points, where each point represents an approximation of the solution $u$ at its corresponding coordinates.

Assuming a square domain $[0, L] \times [0, L]$, we discretize it with $(N_x + 2, N_y + 2)$ points in each dimension, leading to grid spacings defined as $h_x = \frac{L}{N_x + 1}$ and $h_y = \frac{L}{N_y + 1}$.
Let $u_{i,j}$ denote the approximation of $u(x_i, y_j)$ at the grid point $(x_i, y_j) = (ih, jh)$.

The 2D Laplacian can be approximated at each of the $N^2$ interior grid points $(i, j)$ using the central difference formula:

```math
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}
```

```math
\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}
```

Combining these approximations yields the discrete form of the Poisson equation, which can be expressed as:

```math
\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = f_{i,j}
```

This results in a system of linear equations for the unknowns $u_{i,j}$ at each interior grid point.

### Boundary conditions

To complete the system, we need to define boundary conditions for $u$ along the edges of the domain. Common options include:

- **Dirichlet boundary conditions**: These specify the value of $u$ directly on the boundary.
- **Neumann boundary conditions**: These specify the derivative (or flux) of $u$ normal to the boundary.

### Implementing halo regions with `MyVector`

In practical applications, particularly in parallel computing, it is common to introduce **halo regions** (or ghost cells) around the grid.
These additional layers store boundary values from neighboring subdomains, allowing each subdomain to compute stencils near its boundaries independently, without the need for immediate communication with neighboring domains. Halo regions thus simplify boundary condition management in distributed or multi-threaded environments.

In the context of Krylov.jl, while internal storage for each Krylov method typically expects an `AbstractVector`, specific applications can benefit from structured data layouts.
This is where a specialized vector type called **`MyVector`** becomes advantageous.

By using `MyVector` with halo regions, we can implement finite difference stencils without the overhead of boundary condition checks.
This not only enhances code readability but also improves performance.
The type **`OffsetArray`** from the package [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl) supports custom indexing, making it ideal for grids that include halo regions.
By wrapping an `OffsetArray` within a `MyVector`, we can access elements using custom offsets that align with the physical layout of the grid.
This configuration enables **"if-less"** stencils, effectively avoiding direct boundary condition checks within the core loop, resulting in cleaner and potentially faster code.

Moreover, the design of `MyVector` is flexible and can be easily adapted for 1D, 2D, or 3D problems with minimal changes, providing versatility in handling various grid configurations.

### Definition of the `MyVector`

The `MyVector` type is a specialized vector designed for efficient handling of grid-based computations, particularly in the context of finite difference methods with halo regions.
It is parameterized by:

- **`FC`**: The element type of the vector.
- **`D`**: The type of the data array, which utilizes `OffsetArray` to facilitate custom indexing.

Below is the definition of `MyVector`:

```julia
using OffsetArrays

struct MyVector{FC, D} <: AbstractVector{FC}
    data::D

    function MyVector(data::D) where {D}
        FC = eltype(data)
        return new{FC, D}(data)
    end
end

function MyVector{FC,D}(::UndefInitializer, l::Int64) where {FC,D}
    m = n = sqrt(l) |> Int
    data = zeros(FC, m + 2, n + 2)
    v = OffsetMatrix(data, 0:m + 1, 0:n + 1)
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
    row = div(idx - 1, n - 2) + 1
    col = mod(idx - 1, n - 2) + 1
    return v.data[row, col]
end
```

The `size` and `getindex` functions are defined to facilitate display in the REPL, allowing for easy interaction with `MyVector`, although they are not strictly necessary for the functionality of Krylov.jl.

### Stencil implementation

To efficiently apply the discrete Laplacian operator using a matrix-free approach and a typical 5-point stencil operation, we leverage `OffsetArray` for handling halo regions.
This allows seamless access to boundary values, enabling clear and performant computation of the Laplacian without the need for direct boundary condition checks within the core stencil loop.

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

### Required methods for Krylov.jl compatibility

To use `MyVector` within Krylov.jl, we must define several essential vector operations such as dot products, norms, scalar multiplication, and element-wise updates.
Below, we present the required methods that ensure `MyVector` is compatible with Krylov.jl:

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

These methods enable Krylov.jl to utilize custom vector types, enhancing the flexibility and performance of Krylov solvers.

### Solve the 2D poisson equation

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

### Conclusion

The implementation of a 2D Poisson equation solver using `MyVector` offers several notable advantages that enhance both code clarity and efficiency.
With the custom indexing capabilities of `OffsetArray`, we can seamlessly incorporate boundary data from halo regions, simplifying the code by eliminating boundary checks within the core loop.
This results in a more readable and maintainable code structure.

Moreover, by removing boundary checks, we reduce branching in the code, which improves computational efficiency, especially for large grids.
This approach leads to faster execution times while preserving optimal performance.
The flexibility of `MyVector` also makes it easy to extend for more complex stencils or additional dimensions, such as 3D grids.

!!! info
    The package [Oceanigans.jl](https://github.com/CliMA/Oceananigans.jl) utilizes a similar approach with its type `Field` to solve linear systems involving millions of variables efficiently with Krylov.jl.
