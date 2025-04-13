mutable struct StorageGetxRestartedGmres{S}
  x::S
  y::S
  p::S
end
StorageGetxRestartedGmres(workspace::GmresWorkspace; N = I) =
  StorageGetxRestartedGmres(similar(workspace.x), similar(workspace.z), (N === I) ? similar(workspace.p) : similar(workspace.x))

function get_x_restarted_gmres!(workspace::GmresWorkspace{T,FC,S}, A,
                                stor::StorageGetxRestartedGmres{S}, N) where {T,FC,S}
  NisI = (N === I)
  x2, y2, p2 = stor.x, stor.y, stor.p
  n = size(A, 2)
  # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
  nr = sum(1:workspace.inner_iter)
  y = workspace.z  # yᵢ = zᵢ
  y2 .= y
  R = workspace.R
  V = workspace.V
  x2 .= workspace.Δx
  for i = workspace.inner_iter : -1 : 1
    pos = nr + i - workspace.inner_iter      # position of rᵢ.ₖ
    for j = workspace.inner_iter : -1 : i+1
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
  for i = 1 : workspace.inner_iter
    Krylov.kaxpy!(n, y2[i], V[i], x2)
  end
  if !NisI
    p2 .= workspace.p
    p2 .= x2
    mul!(x2, N, p2)
  end
  x2 .+= workspace.x
end

mutable struct TestCallbackN2{T, S, M}
  A::M
  b::S
  storage_vec::S
  tol::T
end
TestCallbackN2(A, b; tol = 0.1) = TestCallbackN2(A, b, similar(b), tol)

function (cb_n2::TestCallbackN2)(solver)
  mul!(cb_n2.storage_vec, cb_n2.A, workspace.x)
  cb_n2.storage_vec .-= cb_n2.b
  return norm(cb_n2.storage_vec) ≤ cb_n2.tol
end

mutable struct TestCallbackN2Adjoint{T, S, M}
  A::M
  b::S
  c::S
  storage_vec1::S
  storage_vec2::S
  tol::T
end
TestCallbackN2Adjoint(A, b, c; tol = 0.1) = TestCallbackN2Adjoint(A, b, c, similar(b), similar(c), tol)

function (cb_n2::TestCallbackN2Adjoint)(solver)
  mul!(cb_n2.storage_vec1, cb_n2.A, workspace.x)
  cb_n2.storage_vec1 .-= cb_n2.b
  mul!(cb_n2.storage_vec2, cb_n2.A', workspace.y)
  cb_n2.storage_vec2 .-= cb_n2.c
  return (norm(cb_n2.storage_vec1) ≤ cb_n2.tol && norm(cb_n2.storage_vec2) ≤ cb_n2.tol)
end

mutable struct TestCallbackN2Shifts{T, S, M}
  A::M
  b::S
  shifts::Vector{T}
  tol::T
end
TestCallbackN2Shifts(A, b, shifts; tol = 0.1) = TestCallbackN2Shifts(A, b, shifts, tol)

function (cb_n2::TestCallbackN2Shifts)(solver)
  r = residuals(cb_n2.A, cb_n2.b, cb_n2.shifts, workspace.x)
  return all(map(norm, r) .≤ cb_n2.tol)
end

mutable struct TestCallbackN2LS{T, S, M}
  A::M
  b::S
  λ::T
  storage_vec1::S
  storage_vec2::S
  tol::T
end
TestCallbackN2LS(A, b, λ; tol = 0.1) = TestCallbackN2LS(A, b, λ, similar(b), similar(b, size(A, 2)), tol)

function (cb_n2::TestCallbackN2LS)(solver)
  mul!(cb_n2.storage_vec1, cb_n2.A, workspace.x)
  cb_n2.storage_vec1 .-= cb_n2.b
  mul!(cb_n2.storage_vec2, cb_n2.A', cb_n2.storage_vec1)
  cb_n2.storage_vec2 .+= cb_n2.λ .* workspace.x
  return norm(cb_n2.storage_vec2) ≤ cb_n2.tol
end

mutable struct TestCallbackN2LN{T, S, M}
  A::M
  b::S
  λ::T
  storage_vec::S
  tol::T
end
TestCallbackN2LN(A, b, λ; tol = 0.1) = TestCallbackN2LN(A, b, λ, similar(b), tol)

function (cb_n2::TestCallbackN2LN)(solver)
  mul!(cb_n2.storage_vec, cb_n2.A, workspace.x)
  cb_n2.storage_vec .-= cb_n2.b
  cb_n2.λ != 0 && (cb_n2.storage_vec .+= cb_n2.λ .* workspace.x)
  return norm(cb_n2.storage_vec) ≤ cb_n2.tol
end

mutable struct TestCallbackN2SaddlePts{T, S, M}
  A::M
  b::S
  c::S
  storage_vec1::S
  storage_vec2::S
  tol::T
end
TestCallbackN2SaddlePts(A, b, c; tol = 0.1) = 
  TestCallbackN2SaddlePts(A, b, c, similar(b), similar(c), tol)

function (cb_n2::TestCallbackN2SaddlePts)(solver)
  mul!(cb_n2.storage_vec1, cb_n2.A, workspace.y)
  cb_n2.storage_vec1 .+= workspace.x .- cb_n2.b
  mul!(cb_n2.storage_vec2, cb_n2.A', workspace.x)
  cb_n2.storage_vec2 .-= workspace.y .+ cb_n2.c
  return (norm(cb_n2.storage_vec1) ≤ cb_n2.tol && norm(cb_n2.storage_vec2) ≤ cb_n2.tol)
end

function restarted_gmres_callback_n2(workspace::GmresWorkspace, A, b, stor, N, storage_vec, tol)
  get_x_restarted_gmres!(solver, A, stor, N)
  x = stor.x
  mul!(storage_vec, A, x)
  storage_vec .-= b
  return (norm(storage_vec) ≤ tol)
end

mutable struct TestCallbackN2LSShifts{T, S, M}
  A::M
  b::S
  shifts::Vector{T}
  tol::T
end
TestCallbackN2LSShifts(A, b, shifts; tol = 0.1) = TestCallbackN2LSShifts(A, b, shifts, tol)

function (cb_n2::TestCallbackN2LSShifts)(solver)
  r = residuals_ls(cb_n2.A, cb_n2.b, cb_n2.shifts, workspace.x)
  return all(map(norm, r) .≤ cb_n2.tol)
end
