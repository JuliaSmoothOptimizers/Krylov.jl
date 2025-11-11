# [Custom workspaces](@id custom_workspaces)

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

```@example halo-regions; continued = true
using OffsetArrays

struct HaloVector{FC, D} <: AbstractVector{FC}
    data::D

    function HaloVector(data::D) where {D}
        FC = eltype(data)
        return new{FC, D}(data)
    end
end

function Base.similar(v::HaloVector)
    data = similar(v.data)
    return HaloVector(data)
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

The functions `similar` and `length` are mandatory and must be implemented for custom vector types.
The functions `size` and `getindex` support REPL display, aiding interaction, though they are optional for Krylov.jl’s functionality.

### Efficient stencil implementation

Using `HaloVector` with `OffsetArray`, we can apply the discrete Laplacian operator in a matrix-free approach with a 5-point stencil, managing halo regions effectively.
This layout allows **clean and efficient Laplacian computation** without boundary checks within the core loop.
For the multiplication between the Laplacian operator and vectors, `Krylov.jl` dispatches to `LinearAlgebra.mul!` by default.
If you prefer to use a custom implementation only within `Krylov.jl`, or if you have a specialized routine that benefits repeated products, you can overload `Krylov.kmul!` instead.

```@example halo-regions; continued = true
using Krylov

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

function Krylov.kmul!(y::HaloVector{Float64}, A::LaplacianOperator, u::HaloVector{Float64})
    # Apply the discrete Laplacian in 2D
    for i in 1:A.Nx
        for j in 1:A.Ny
            # Calculate second derivatives using finite differences
            dx2 = (u.data[i-1,j] - 2 * u.data[i,j] + u.data[i+1,j]) / (A.Δx)^2
            dy2 = (u.data[i,j-1] - 2 * u.data[i,j] + u.data[i,j+1]) / (A.Δy)^2
            
            # Update the output vector with the Laplacian result
            y.data[i,j] = dx2 + dy2

            # Add a tiny value ε to the diagonal to ensure the operator is
            # symmetric positive definite (SPD) and to avoid numerical issues
            if i == j
                y.data[i,j] += 1e-12 * u.data[i,j]
            end
        end
    end

    return y
end
```

### Methods to overload for compatibility with Krylov.jl

To integrate `HaloVector` with Krylov.jl, we define essential vector operations, including dot products, norms, scalar multiplication, and element-wise updates.
These implementations allow Krylov.jl to leverage custom vector types, enhancing both solver flexibility and performance.

```@example halo-regions; continued = true
using Krylov
import Krylov.FloatOrComplex

function Krylov.kdot(n::Integer, x::HaloVector{T}, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    res = zero(real(T))
    for j = 1:nx-1
        for i = 1:mx-1
            res += _x[i,j] * conj(_y[i,j])
        end
    end
    return res
end

function Krylov.knorm(n::Integer, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    res = zero(real(T))
    for j = 1:nx-1
        for i = 1:mx-1
            res += abs2(_x[i,j])
        end
    end
    return sqrt(res)
end

function Krylov.kscal!(n::Integer, s::T, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for j = 1:nx-1
        for i = 1:mx-1
            _x[i,j] = s * _x[i,j]
        end
    end
    return x
end

function Krylov.kdiv!(n::Integer, x::HaloVector{T}, s::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for j = 1:nx-1
        for i = 1:mx-1
            _x[i,j] = _x[i,j] / s
        end
    end
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::HaloVector{T}, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            _y[i,j] += s * _x[i,j]
        end
    end
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::HaloVector{T}, t::T, y::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            _y[i,j] = s * _x[i,j] + t * _y[i,j]
        end
    end
    return y
end

function Krylov.kcopy!(n::Integer, y::HaloVector{T}, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            _y[i,j] = _x[i,j]
        end
    end
    return y
end

function Krylov.kscalcopy!(n::Integer, y::HaloVector{T}, s::T, x::HaloVector{T}) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            _y[i,j] = s * _x[i,j]
        end
    end
    return y
end

function Krylov.kdivcopy!(n::Integer, y::HaloVector{T}, x::HaloVector{T}, s::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            _y[i,j] = _x[i,j] / s
        end
    end
    return y
end

function Krylov.kfill!(x::HaloVector{T}, val::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    for j = 1:nx-1
        for i = 1:mx-1
            _x[i,j] = val
        end
    end
    return x
end

function Krylov.kref!(n::Integer, x::HaloVector{T}, y::HaloVector{T}, c::T, s::T) where T <: FloatOrComplex
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for j = 1:nx-1
        for i = 1:mx-1
            x_ij = _x[i,j]
            y_ij = _y[i,j]
            _x[i,j] = c       * x_ij + s * y_ij
            _y[i,j] = conj(s) * x_ij - c * y_ij
        end
    end
    return x, y
end
```

By default, `kdiv!(n, y, x, s)` calls `kscal!(n, y, t, x)` with `t = 1/s`, so a separate implementation isn't required.
However, this approach may introduce numerical issues when `s` is very small.
We do this because computing $y \leftarrow t \times x$ can often leverage SIMD or fused multiply-add (FMA) instructions on certain architectures, capabilities that a direct element-wise division $y \leftarrow x/s$ typically lacks.
Thus, the implementation of `kdiv!` provides flexibility, allowing users to choose a trade-off between speed and numerical precision by overloading the function if needed.
The operations provided by `kdivcopy!` and `kscalcopy!` could be implemented directly by using `kcopy!`, `kscal!`, and `kdiv!` but require two separate memory passes, which can be suboptimal for performance.
To address this limitation, `kdivcopy!` and `kscalcopy!` fuse the copy and scaling/division operations into a single memory pass.
Note that `Krylov.kref!` is only required for the function `minres_qlp`.

### 2D Poisson equation solver with Krylov methods

```@example halo-regions
using Krylov, OffsetArrays
import Krylov: solution, statistics

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
for j in 1:Ny
    for i in 1:Nx
        xi = i * Δx
        yj = j * Δy
        data[i,j] = f(xi, yj)
    end
end
b = HaloVector(data)

# Allocate the workspace
kc = KrylovConstructor(b)
workspace = CgWorkspace(kc)

# Solve the system with CG
Krylov.cg!(workspace, A, b, atol=1e-10, rtol=0.0, verbose=1)
u_sol = solution(workspace)
stats = statistics(workspace)
```

```@example halo-regions
# The exact solution is u(x,y) = sin(πx) * sin(πy)
u_star = [sin(π * i * Δx) * sin(π * j * Δy) for i=1:Nx, j=1:Ny]
norm(u_sol.data[1:Nx, 1:Ny] - u_star, Inf)
```

!!! note
    Only the in-place version of the Krylov methods is supported for custom vector types.

### Conclusion

Implementing a 2D Poisson equation solver with `HaloVector` improves code clarity and efficiency.
Custom indexing with `OffsetArray` streamlines halo region management, eliminating boundary checks within the core loop.
This approach reduces branching, yielding faster execution, especially on large grids.
`HaloVector`'s flexibility also makes it easy to extend to 3D grids or more complex stencils.

!!! info
    [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) uses a similar strategy with its type `KrylovField`, efficiently solving large linear systems with Krylov.jl.

## Solving saddle point systems with BlockArrays.jl and Krylov.jl

[BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) simplifies working with structured matrices, making it an ideal tool for solving saddle point systems.
In this example, we solve a structured linear system $Ax = b$ of the form:

```math
\begin{bmatrix}
   K & B^T \\
   B & 0
\end{bmatrix}
\begin{bmatrix}
   y \\
   z
\end{bmatrix} =
\begin{bmatrix}
   c \\
   d
\end{bmatrix}.
```

We first define the matrix $A$ and the vector $b$ using `BlockArrays.jl`:

```@example block-arrays; continued = true
using LinearAlgebra
using BlockArrays

nK = 10
nB = 2
K = rand(nK, nK)
K = K * K' + I
B = rand(nB, nK)
c = rand(nK)
d = rand(nB)

# Create the saddle point matrix A
A = BlockArray{Float64}(undef, [nK, nB], [nK, nB])
A[Block(1, 1)] = K
A[Block(1, 2)] = B'
A[Block(2, 1)] = B
A[Block(2, 2)] = zeros(nB, nB)

# Create the right-hand side vector b
b = BlockVector{Float64}(undef, [nK, nB])
b[Block(1)] = c
b[Block(2)] = d
```

For saddle point systems, a well-known preconditioner is the "ideal preconditioner'' $P^{-1}$, as described in the paper ["A Note on Preconditioning for Indefinite Linear Systems"](https://doi.org/10.1137/S1064827599355153).
It is defined as:

```math
P^{-1} =
\begin{bmatrix}
   K^{-1} & 0
\\ 0      & (B K^{-1} B^T)^{-1}
\end{bmatrix}.
```

This preconditioner guarantees convergence in exactly three iterations, as $P^{-1}A$ has only three distinct eigenvalues.
However, this preconditioner is expensive, as it requires $K^{-1}$ and the inverse of the Schur complement $B K^{-1} B^T$.
One common approach is to replace $K^{-1}$ with $\mathrm{diag}(K)^{-1}$, creating a cheaper preconditioner.

```@example block-arrays; continued = true
using Krylov
using LinearAlgebra

struct IdealPreconditioner{T1, T2}
    BD1::T1
    BD2::T2
end

function Krylov.kmul!(y::BlockVector, P::IdealPreconditioner, x::BlockVector)
    mul!(y.blocks[1], P.BD1, x.blocks[1])
    mul!(y.blocks[2], P.BD2, x.blocks[2])
    return y
end

# Create the ideal preconditioner
BD1 = inv(K)
BD2 = inv(B * BD1 * B')
P = IdealPreconditioner(BD1, BD2)
```

We now solve the system $Ax = b$ using `minres` with our preconditioner:

```@example block-arrays
using Krylov
import Krylov: solution, iteration_count

kc = KrylovConstructor(b)
workspace = MinresWorkspace(kc)
minres!(workspace, A, b; M=P)

x = solution(workspace)
niter = iteration_count(workspace)
```

This example demonstrates how `BlockArrays.jl` and `Krylov.jl` can be effectively combined to solve structured saddle point systems.

## Distributed Krylov solvers with MPI.jl

Krylov.jl supports distributed-memory computations through [MPI.jl](https://github.com/JuliaParallel/MPI.jl), enabling Krylov solvers to scale across multiple processes and compute nodes.
This makes them suitable for solving large-scale linear systems in high-performance computing (HPC) environments.

To get started, you need to install the `mpiexecjl` binary using the function `MPI.install_mpiexecjl`.
If you wish to change the underlying MPI implementation, you can use the package `MPIPreferences.jl` and its functions `MPIPreferences.use_binary_binary` and `MPIPreferences.use_jll_binary`.

Next, define an `MPIVector` to be used as the storage type for Krylov workspaces, replacing the standard `Vector`.

```julia
using MPI

struct MPIVector{T, V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    comm::MPI.Comm
    global_len::Int
    data_range::UnitRange{Int}
end

Base.similar(v::MPIVector) = MPIVector(similar(v.data), v.comm, v.global_len, v.data_range)
Base.length(v::MPIVector) = v.global_len
Base.eltype(v::MPIVector{T}) where T = T

function MPIVector(global_vec::V, comm::MPI.Comm = MPI.COMM_WORLD) where V
    T = eltype(V)
    size, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    n = length(global_vec)
    chunk = div(n + size - 1, size)
    istart = rank * chunk + 1
    iend = min((rank + 1) * chunk, n)
    data = global_vec[istart:iend]
    return MPIVector{T, V}(data, comm, n, istart:iend)
end

function MPIVector(::Type{V}, global_len::Int, comm::MPI.Comm = MPI.COMM_WORLD) where V
    size, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    chunk = div(global_len + size - 1, size)
    istart = rank * chunk + 1
    iend = min((rank + 1) * chunk, global_len)
    data = V(undef, iend-istart+1)
    return MPIVector{T, V}(data, comm, global_len, istart:iend)
end
```

To integrate `MPIVector` with Krylov.jl, you need to define the essential vector operations.
These implementations enable Krylov.jl to operate with MPI-based vectors on both CPU and GPU.

```julia
using Krylov
import Krylov.FloatOrComplex


function Krylov.kdot(n::Integer, x::MPIVector{T}, y::MPIVector{T}) where T <: FloatOrComplex
    data_dot = sum(x.data .* conj.(y.data))
    res = MPI.Allreduce(data_dot, +, x.comm)
    return res
end

function Krylov.knorm(n::Integer, x::MPIVector{T}) where T <: FloatOrComplex
    res = Krylov.kdot(n, x, x)
    return sqrt(res)
end

function Krylov.kscal!(n::Integer, s::T, x::MPIVector{T}) where T <: FloatOrComplex
    x.data .*= s
    return x
end

function Krylov.kdiv!(n::Integer, x::MPIVector{T}, s::T) where T <: FloatOrComplex
    x.data ./= s
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::MPIVector{T}, y::MPIVector{T}) where T <: FloatOrComplex
    y.data .+= s .* x.data
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::MPIVector{T}, t::T, y::MPIVector{T}) where T <: FloatOrComplex
    y.data .= s .* x.data .+ t .* y.data
    return y
end

function Krylov.kcopy!(n::Integer, y::MPIVector{T}, x::MPIVector{T}) where T <: FloatOrComplex
    y.data .= x.data
    return y
end

function Krylov.kscalcopy!(n::Integer, y::MPIVector{T}, s::T, x::MPIVector{T}) where T <: FloatOrComplex
    y.data .= x.data .* s
    return y
end

function Krylov.kdivcopy!(n::Integer, y::MPIVector{T}, x::MPIVector{T}, s::T) where T <: FloatOrComplex
    y.data .= x.data ./ s
    return y
end

function Krylov.kfill!(x::MPIVector{T}, val::T) where T <: FloatOrComplex
    x.data .= val
    return x
end
```

The next step is to create a distributed linear operator.
As a simple example, we define a diagonal operator with `1:n` on its diagonal by implementing an `MPIOperator`:

```julia
using Krylov

struct MPIOperator{T}
    m::Int
    n::Int
end

Base.size(A::MPIOperator) = (A.m, A.n)
Base.eltype(A::MPIOperator{T}) where T = T

function Krylov.kmul!(y::MPIVector{Float64}, A::MPIOperator{Float64}, u::MPIVector{Float64})
    y.data .= u.data_range .* u.data
    return y
end
```

With all the previous definitions in place, you can now assemble a script `mpi_krylov.jl` that includes them, along with a small driver code provided below.
This script can be executed with multiple processes using the binary `mpiexecjl`.

```julia
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n = 10  # global_len in MPIVector
T = Float64
method = :cg

# Version without MPI
A_global = Diagonal(1:n) .|> T
b_global = collect(n:-1:1) .|> T
x_global, _ = krylov_solve(Val(method), A_global, b_global)

# Version with MPI
A_mpi = MPIOperator{T}(n, n)
b_mpi = MPIVector(b_global)

kc = KrylovConstructor(b_mpi)
workspace = krylov_workspace(Val(method), kc)
krylov_solve!(workspace, A_mpi, b_mpi)
x_mpi = Krylov.solution(workspace)

# Check the solution
for i = 0:MPI.Comm_size(comm)-1
    if i == rank
        println("Local data: ", x_mpi.data)
        println("Global data: ", x_global)
    end
    MPI.Barrier(comm)
end

MPI.Finalize()
```

To launch this script on four processes:
```shell
mpiexecjl -n 4 julia mpi_krylov.jl
```

The example above runs on CPU but also works on GPU.
It can target supercomputers such as Frontier or Aurora with various GPU backends (NVIDIA, AMD, or Intel).
See the section on [GPU support](@ref gpu) for details on the GPU-enabled vectors usable in an `MPIVector`.

!!! note
    Make sure to use an MPI implementation that is GPU-aware if you intend to run distributed Krylov solvers on GPUs.
