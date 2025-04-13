# [Warm-start](@id warm-start)

Most Krylov methods in this module accept a starting point as argument.
The starting point is used as initial approximation to a solution.

```julia
workspace = CgWorkspace(A, b)
cg!(workspace, A, b, itmax=100)
if !issolved(solver)
  cg!(workspace, A, b, workspace.x, itmax=100) # cg! uses the approximate solution `workspace.x` as starting point
end
```

If the user has an initial guess `x0`, it can be provided directly.

```julia
cg(A, b, x0)
```

It is also possible to use the `warm_start!` function to feed the starting point into the workspace.

```julia
warm_start!(workspace, x0)
cg!(workspace, A, b)
# the previous two lines are equivalent to cg!(workspace, A, b, x0)
```

If a Krylov method doesn't have the option to warm start, it can still be done explicitly.
We provide an example with `cg_lanczos!`.

```julia
workspace = CgLanczosWorkspace(A, b)
cg_lanczos!(workspace, A, b)
x₀ = workspace.x           # Ax₀ ≈ b
r = b - A * x₀          # r = b - Ax₀
cg_lanczos!(workspace, A, r)
Δx = workspace.x           # AΔx = r
x = x₀ + Δx             # Ax = b
```

Explicit restarts cannot be avoided in certain block methods, such as TriMR, due to the preconditioners.

```julia
# [E  A] [x] = [b]
# [Aᴴ F] [y]   [c]
M = inv(E)
N = inv(F)
x₀, y₀, stats = trimr(A, b, c, M=M, N=N)

# E and F are not available inside TriMR
b₀ = b -  Ex₀ - Ay
c₀ = c - Aᴴx₀ - Fy

Δx, Δy, stats = trimr(A, b₀, c₀, M=M, N=N)
x = x₀ + Δx
y = y₀ + Δy
```
```@meta
# ## Restarted methods
#
# The storage requirements of Krylov methods based on the Arnoldi process, such as FOM and GMRES, increase as the iteration progresses.
# For very large problems, the storage costs become prohibitive after only few iterations and restarted variants FOM(k) and GMRES(k) are preferred.
# In this section, we show how to use warm starts to implement GMRES(k) and FOM(k).
#
# ```julia
# k = 50
# workspace = GmresWorkspace(A, b, k)  # FomWorkspace(A, b, k)
# workspace.x .= 0                  # workspace.x .= x₀ 
# nrestart = 0
# while !issolved(solver) || nrestart ≤ 10
#   solve!(workspace, A, b, workspace.x, itmax=k)
#   nrestart += 1
# end
# ```
```
