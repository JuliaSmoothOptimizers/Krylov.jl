## Callbacks

Each Krylov method is able to use a callback function that is called as `callback(solver)`, and that should return `true` if the main loop should terminate, and `false` otherwise.
If the method terminated because of the callback, the output status will be `"user-requested exit"`.
For example, if the user has a function `my_callback(solver::MinresSolver)`, it can be used with:

```julia
(x, stats) = minres(A, b, callback = my_callback)
```

If you need to write a callback that uses variables that are not in the `MinresSolver`, it is possible to use an anonymous function:

```julia
function my_callback2(solver::MinresSolver, A, b, storage_vec, tol::Float64)
  mul!(storage_vec, A, solver.x)
  storage_vec .-= b
  return norm(storage_vec) ≤ tol # tolerance based on the 2-norm of the residual
end

storage_vec = similar(b)
(x, stats) = minres(A, b, callback = solver -> my_callback2(solver, A, b, storage_vec, 0.1))
```

You can also use a structure and call it as an anonymous function:

```julia
mutable struct MyCallback3{S, M}
  A::M
  b::S
  storage_vec::S
  tol::Float64
end
MyCallback3(A, b; tol = 0.1) = MyCallback3(A, b, similar(b), tol)

function (my_cb::MyCallback3)(solver)
  mul!(my_cb.storage_vec, my_cb.A, solver.x)
  my_cb.storage_vec .-= my_cb.b
  return norm(my_cb.storage_vec) ≤ my_cb.tol # tolerance based on the 2-norm of the residual
end

my_cb = MyCallback3(A, b; tol = 0.1)
(x, stats) = minres(A, b, callback = solver -> my_cb(solver))
```
