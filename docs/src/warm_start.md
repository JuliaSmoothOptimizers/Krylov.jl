## Warm Start

Krylov methods that have the `restart` parameter can use the current state of the solver as its starting point.

```julia
solver = CgSolver(n, n, S)
cg!(solver, A, b, itmax=100)
if !issolved(solver)
  cg!(solver, A, b, restart=true, itmax=100)
end
```

If the user has an initial guess, it can be also provided to the `solver`.

```julia
solver.x .= x₀
cg!(solver, A, b, restart=true)
```

If a Krylov method doesn't have the option `restart`, it can be restarted explicitly.

```julia
solver = BicgstabSolver(n, n, S)
bicgstab!(solver, A, b)
x₀ = copy(solver.x)     # Ax₀ ≈ b
r = b - A * x₀          # r = b - Ax₀
bicgstab!(solver, A, r)
Δx = solver.x           # AΔx = r
x = x₀ + Δx             # Ax = b
```

Explicit restarts can't be avoid in same cases due to the preconditioners.

```julia
# [E  A] [x] = [b]
# [Aᵀ F] [y]   [c]
M = inv(E)
N = inv(F)
x₀, y₀, stats = trimr(A, b, c, M=M, N=N)

# E and F are not available inside TriMR
b₀ = b -  Ex₀ - Ay
c₀ = c - Aᵀx₀ - Fy

Δx, Δy, stats = trimr(A, b₀, c₀, M=M, N=N)
x = x₀ + Δx
y = y₀ + Δy
```

## Restarted methods

The storage requierements of Krylov methods based on the Arnoldi process, such as FOM and GMRES, increase as the iteration progresses.
For very large problems, the storage costs become prohibitive after only few iterations and restarted variants FOM(k) and GMRES(k) are prefered.
In this section, we present how GMRES(k) and FOM(k) can be implemented thanks to the `restart` option.

```julia
k = 50
solver = GmresSolver(A, b, k)  # FomSolver(A, b, k)
solver.x .= 0                  # solver.x .= x₀ 
nrestart = 0
while !issolved(solver) || nrestart ≤ 10
  solve!(solver, A, b, itmax=k, restart=true)
  nrestart += 1
end
```
