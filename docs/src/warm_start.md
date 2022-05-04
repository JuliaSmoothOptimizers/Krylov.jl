## Warm Start

Most Krylov methods in this module accept a starting point as argument. The starting point is used as initial approximation to a solution.

```julia
solver = CgSolver(n, n, S)
cg!(solver, A, b, itmax=100)
if !issolved(solver)
  cg!(solver, A, b, solver.x, itmax=100) # the 4th argument tells cg! to start from solver.x
end
```

If the user has an initial guess `x0`, it can be provided directly.

```julia
cg(A, b, x0)
```

It is also possible to use the `warm_start!` function to feed the starting point into the solver.

```julia
warm_start!(solver, x0)
cg!(solver, A, b)
# the previous two lines are equivalent to cg!(solver, A, b, x0)
```

If a Krylov method doesn't have the option to warm start, it can be still be done explicitly.
We provide an example with `cg_lanczos!`.

```julia
solver = CgLanczosSolver(n, n, S)
cg_lanczos!(solver, A, b)
x₀ = copy(solver.x)     # Ax₀ ≈ b
r = b - A * x₀          # r = b - Ax₀
cg_lanczos!(solver, A, r)
Δx = solver.x           # AΔx = r
x = x₀ + Δx             # Ax = b
```

Explicit restarts cannot be avoided in certain block methods, such as TRIMR, due to the preconditioners.

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
In this section, we show how to use warm starts to implement GMRES(k) and FOM(k).

```julia
k = 50
solver = GmresSolver(A, b, k)  # FomSolver(A, b, k)
solver.x .= 0                  # solver.x .= x₀ 
nrestart = 0
while !issolved(solver) || nrestart ≤ 10
  solve!(solver, A, b, x0, itmax=k)
  nrestart += 1
end
```
