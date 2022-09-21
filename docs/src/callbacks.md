# [Callbacks](@id callbacks)

Each Krylov method is able to call a callback function as `callback(solver)` at each iteration.
The callback should return `true` if the main loop should terminate, and `false` otherwise.
If the method terminated because of the callback, the output status will be `"user-requested exit"`.
For example, if the user defines `minres_callback(solver::MinresSolver)`, it can be passed to the solver using

```julia
(x, stats) = minres(A, b, callback = minres_callback)
```

If you need to write a callback that uses variables that are not in a `KrylovSolver`, use a closure:

```julia
function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
  mul!(r, A, solver.x)
  r .-= b               # r := b - Ax
  bool = norm(r) ≤ tol  # tolerance based on the 2-norm of the residual
  return bool
end

cg_callback(solver) = custom_stopping_condition(solver, A, b, r, tol)
(x, stats) = cg(A, b, callback = cg_callback)
```

Alternatively, use a structure and make it callable:

```julia
mutable struct CallbackWorkspace{T}
  A::Matrix{T}
  b::Vector{T}
  r::Vector{T}
  tol::T
end

function (workspace::CallbackWorkspace)(solver::KrylovSolver)
  mul!(workspace.r, workspace.A, solver.x)
  workspace.r .-= workspace.b
  bool = norm(workspace.r) ≤ workspace.tol
  return bool
end

bicgstab_callback = CallbackWorkspace(A, b, r, tol)
(x, stats) = bicgstab(A, b, callback = bicgstab_callback)
```

Although the main goal of a callback is to add new stopping conditions, it can also retrieve information from the workspace of a Krylov method along the iterations.
We now illustrate how to store all iterates $x_k$ of the GMRES method.

```julia
S = Krylov.ktypeof(b)
global X = S[]  # Storage for GMRES iterates

function gmres_callback(solver)
  z = solver.z
  k = solver.inner_iter
  nr = sum(1:k)
  V = solver.V
  R = solver.R
  y = copy(z)

  # Solve Rk * yk = zk
  for i = k : -1 : 1
    pos = nr + i - k
    for j = k : -1 : i+1
      y[i] = y[i] - R[pos] * y[j]
      pos = pos - j + 1
    end
    y[i] = y[i] / R[pos]
  end

  # xk = Vk * yk
  xk = sum(V[i] * y[i] for i = 1:k)
  push!(X, xk)

  return false  # We don't want to add new stopping conditions
end

(x, stats) = gmres(A, b, callback = gmres_callback)
```
