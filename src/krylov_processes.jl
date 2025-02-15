export hermitian_lanczos, nonhermitian_lanczos, arnoldi, golub_kahan, saunders_simon_yip, montoison_orban

function nonhermitian_lanczos(args...; kwargs...)
  error("The function `nonhermitian_lanczos` requires the package SparseArrays. Add `using SparseArrays` to your code.\nIf SparseArrays is already loaded, check the provided arguments.")
end

function hermitian_lanczos(args...; kwargs...)
  error("The function `hermitian_lanczos` requires the package SparseArrays. Add `using SparseArrays` to your code.\nIf SparseArrays is already loaded, check the provided arguments.")
end

function golub_kahan(args...; kwargs...)
  error("The function `golub_kahan` requires the package SparseArrays. Add `using SparseArrays` to your code.\nIf SparseArrays is already loaded, check the provided arguments.")
end

function saunders_simon_yip(args...; kwargs...)
  error("The function `saunders_simon_yip` requires the package SparseArrays. Add `using SparseArrays` to your code.\nIf SparseArrays is already loaded, check the provided arguments.")
end

"""
    V, β, H = arnoldi(A, b, k; allow_breakdown=false, reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `b`: a vector of length `n`;
* `k`: the number of iterations of the Arnoldi process.

#### Keyword arguments

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs;
* `reorthogonalization`: reorthogonalize each newly added vector of the Krylov basis against all previous vectors (full reorthogonalization).

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `H`: a dense `(k+1) × k` upper Hessenberg matrix.

#### Reference

* W. E. Arnoldi, [*The principle of minimized iterations in the solution of the matrix eigenvalue problem*](https://doi.org/10.1090/qam/42792), Quarterly of Applied Mathematics, 9, pp. 17--29, 1951.
"""
function arnoldi(A, b::AbstractVector{FC}, k::Int;
                 allow_breakdown::Bool=false, reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  β = zero(R)
  V = M(undef, n, k+1)
  H = zeros(FC, k+1, k)

  for j = 1:k
    vⱼ = view(V,:,j)
    vⱼ₊₁ = q = view(V,:,j+1)
    if j == 1
      β = knorm(n, b)
      if β == 0
        !allow_breakdown && error("Exact breakdown β == 0.")
        kfill!(vⱼ, zero(FC))
      else
        vⱼ .= b ./ β
      end
    end
    mul!(q, A, vⱼ)
    for i = 1:j
      vᵢ = view(V,:,i)
      H[i,j] = kdot(n, vᵢ, q)
      kaxpy!(n, -H[i,j], vᵢ, q)
    end
    if reorthogonalization
      for i = 1:j
        vᵢ = view(V,:,i)
        Htmp = kdot(n, vᵢ, q)
        kaxpy!(n, -Htmp, vᵢ, q)
        H[i,j] += Htmp
      end
    end
    H[j+1,j] = knorm(n, q)
    if H[j+1,j] == 0
      !allow_breakdown && error("Exact breakdown Hᵢ₊₁.ᵢ == 0 at iteration i = $j.")
      kfill!(vⱼ₊₁, zero(FC))
    else
      vⱼ₊₁ .= q ./ H[j+1,j]
    end
  end
  return V, β, H
end

"""
    V, β, H, U, γ, F = montoison_orban(A, B, b, c, k; allow_breakdown=false, reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a linear operator that models a matrix of dimension `n × m`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`;
* `k`: the number of iterations of the Montoison-Orban process.

#### Keyword arguments

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs;
* `reorthogonalization`: reorthogonalize each newly added vector of the Krylov basis against all previous vectors (full reorthogonalization).

#### Output arguments

* `V`: a dense `m × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `H`: a dense `(k+1) × k` upper Hessenberg matrix;
* `U`: a dense `n × (k+1)` matrix;
* `γ`: a coefficient such that `γu₁ = c`;
* `F`: a dense `(k+1) × k` upper Hessenberg matrix.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://doi.org/10.1137/21M1459265), SIAM Journal on Matrix Analysis and Applications, 44(1), pp. 293--311, 2023.
"""
function montoison_orban(A, B, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
                         allow_breakdown::Bool=false, reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  β = γ = zero(R)
  V = M(undef, m, k+1)
  U = M(undef, n, k+1)
  H = zeros(FC, k+1, k)
  F = zeros(FC, k+1, k)

  for j = 1:k
    vⱼ = view(V,:,j)
    uⱼ = view(U,:,j)
    vⱼ₊₁ = q = view(V,:,j+1)
    uⱼ₊₁ = p = view(U,:,j+1)
    if j == 1
      β = knorm(m, b)
      if β == 0
        !allow_breakdown && error("Exact breakdown β == 0.")
        kfill!(vⱼ, zero(FC))
      else
        vⱼ .= b ./ β
      end
      γ = knorm(n, c)
      if γ == 0
        !allow_breakdown && error("Exact breakdown γ == 0.")
        kfill!(uⱼ, zero(FC))
      else
       uⱼ .= c ./ γ
      end
    end
    mul!(q, A, uⱼ)
    mul!(p, B, vⱼ)
    for i = 1:j
      vᵢ = view(V,:,i)
      uᵢ = view(U,:,i)
      H[i,j] = kdot(m, vᵢ, q)
      kaxpy!(n, -H[i,j], vᵢ, q)
      F[i,j] = kdot(n, uᵢ, p)
      kaxpy!(m, -F[i,j], uᵢ, p)
    end
    if reorthogonalization
      for i = 1:j
        vᵢ = view(V,:,i)
        uᵢ = view(U,:,i)
        Htmp = kdot(m, vᵢ, q)
        kaxpy!(m, -Htmp, vᵢ, q)
        H[i,j] += Htmp
        Ftmp = kdot(n, uᵢ, p)
        kaxpy!(n, -Ftmp, uᵢ, p)
        F[i,j] += Ftmp
      end
    end
    H[j+1,j] = knorm(m, q)
    if H[j+1,j] == 0
      !allow_breakdown && error("Exact breakdown Hᵢ₊₁.ᵢ == 0 at iteration i = $j.")
      kfill!(vⱼ₊₁, zero(FC))
    else
      vⱼ₊₁ .= q ./ H[j+1,j]
    end
    F[j+1,j] = knorm(n, p)
    if F[j+1,j] == 0
      !allow_breakdown && error("Exact breakdown Fᵢ₊₁.ᵢ == 0 at iteration i = $j.")
      kfill!(uⱼ₊₁, zero(FC))
    else
      uⱼ₊₁ .= p ./ F[j+1,j]
    end
  end
  return V, β, H, U, γ, F
end
