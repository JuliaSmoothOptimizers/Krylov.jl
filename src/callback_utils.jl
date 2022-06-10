export StorageGetxRestartedGmres

export get_x_restarted_gmres!

mutable struct StorageGetxRestartedGmres{S}
  x::S
  y::S
  p::S
end
StorageGetxRestartedGmres(solver::GmresSolver; N = I) = 
  StorageGetxRestartedGmres(similar(solver.x), similar(solver.z), (N === I) ? similar(solver.p) : similar(solver.x))

function get_x_restarted_gmres!(solver::GmresSolver{T,FC,S}, A, 
                                stor::StorageGetxRestartedGmres{S}, N) where {T,FC,S}
  NisI = (N === I)
  x2, y2, p2 = stor.x, stor.y, stor.p
  n = size(A, 2)
  # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
  nr = sum(1:solver.inner_iter)
  y = solver.z  # yᵢ = zᵢ
  y2 .= y
  R = solver.R
  V = solver.V
  x2 .= solver.Δx
  for i = solver.inner_iter : -1 : 1
    pos = nr + i - solver.inner_iter      # position of rᵢ.ₖ
    for j = solver.inner_iter : -1 : i+1
      y2[i] = y2[i] - R[pos] * y2[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
      pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
    end
    # Rₖ can be singular if the system is inconsistent
    if abs(R[pos]) ≤ eps(T)^(3/4)
      y2[i] = zero(FC)
      inconsistent = true
    else
      y2[i] = y2[i] / R[pos]  # yᵢ ← yᵢ / rᵢᵢ
    end
  end

  # Form xₖ = N⁻¹Vₖyₖ
  for i = 1 : solver.inner_iter
    @kaxpy!(n, y2[i], V[i], x2)
  end
  if !NisI
    p2 .= solver.p
    p2 .= x2
    mul!(x2, N, p2)
  end
  x2 .+= solver.x
end