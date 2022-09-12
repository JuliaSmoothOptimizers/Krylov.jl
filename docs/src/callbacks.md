# [Callbacks](@id callbacks)

Each Krylov method is able to call a callback function as `callback(solver)` at each iteration.
The callback should return `true` if the main loop should terminate, and `false` otherwise.
If the method terminated because of the callback, the output status will be `"user-requested exit"`.
For example, if the user defines `my_callback(solver::MinresSolver)`, it can be passed to the solver using

```julia
(x, stats) = minres(A, b, callback = my_callback)
```

If you need to write a callback that uses variables that are not in the `MinresSolver`, use a closure:

```julia
function my_callback2(solver::MinresSolver, A, b, r, tol)
  mul!(r, A, solver.x)
  r .-= b               # r := b - Ax
  bool = norm(r) ≤ tol  # tolerance based on the 2-norm of the residual
  return bool
end

r = similar(b)
(x, stats) = minres(A, b, callback = solver -> my_callback2(solver, A, b, r, 1e-6))
```

Alternatively, use a structure and make it callable:

```julia
mutable struct my_callback3{S, M}
  A::M
  b::S
  r::S
  tol::Float64
end

my_callback3(A, b; tol=1e-6) = my_callback3(A, b, similar(b), tol)  # Outer constructor

function (my_cb::my_callback3)(solver)
  mul!(my_cb.r, my_cb.A, solver.x)
  my_cb.r .-= my_cb.b
  bool = norm(my_cb.r) ≤ my_cb.tol
  return bool
end

my_cb = my_callback3(A, b)
(x, stats) = minres(A, b, callback = my_cb)
```

Although the main goal of a callback is to add new stopping conditions, it can also retrieve informations from the workspace of a Krylov method along the iterations.
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
