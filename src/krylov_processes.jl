export hermitian_lanczos, nonhermitian_lanczos, arnoldi, golub_kahan, saunders_simon_yip, montoison_orban

"""
    V, β, T = hermitian_lanczos(A, b, k; allow_breakdown=false, reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `b`: a vector of length `n`;
* `k`: the number of iterations of the Hermitian Lanczos process.

#### Keyword arguments

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs;
* `reorthogonalization`: reorthogonalize each newly added vector of the Krylov basis against only the two previous vectors (local reorthogonalization).

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix.

#### References

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
* H. D. Simon, [*The Lanczos algorithm with partial reorthogonalization*](https://doi.org/10.1090/S0025-5718-1984-0725988-X), Mathematics of computation, 42(165), pp. 115--142, 1984.
"""
function hermitian_lanczos(A, b::AbstractVector{FC}, k::Int;
                           allow_breakdown::Bool=false, reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval = zeros(R, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = zero(R)
  V = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval)

  pαᵢ = 1  # Position of αᵢ in the vector `nzval`
  for i = 1:k
    vᵢ = view(V,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    if i == 1
      β₁ = knorm(n, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        kdivcopy!(n, vᵢ, b, β₁)
      end
    end
    kmul!(q, A, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      βᵢ = nzval[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      nzval[pαᵢ-1] = βᵢ  # Tᵢ₋₁.ᵢ = βᵢ
      kaxpy!(n, -βᵢ, vᵢ₋₁, q)
    end
    αᵢ = kdotr(n, vᵢ, q)
    kaxpy!(n, -αᵢ, vᵢ, q)
    if reorthogonalization
      if i ≥ 2
        vᵢ₋₁ = view(V,:,i-1)
        βtmp = kdotr(n, vᵢ₋₁, q)
        nzval[pαᵢ-2] += βtmp
        nzval[pαᵢ-1] += βtmp
        kaxpy!(n, -βtmp, vᵢ₋₁, q)
      end
      αtmp = kdotr(n, vᵢ, q)
      αᵢ += αtmp
      kaxpy!(n, -αtmp, vᵢ, q)
    end
    nzval[pαᵢ] = αᵢ  # Tᵢ.ᵢ = αᵢ
    βᵢ₊₁ = knorm(n, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      kdivcopy!(n, vᵢ₊₁, q, βᵢ₊₁)
    end
    nzval[pαᵢ+1] = βᵢ₊₁  # Tᵢ₊₁.ᵢ = βᵢ₊₁
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T
end

"""
    V, β, T, U, γᴴ, Tᴴ = nonhermitian_lanczos(A, b, c, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `b`: a vector of length `n`;
* `c`: a vector of length `n`;
* `k`: the number of iterations of the non-Hermitian Lanczos process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix;
* `U`: a dense `n × (k+1)` matrix;
* `γᴴ`: a coefficient such that `γᴴu₁ = c`;
* `Tᴴ`: a sparse `(k+1) × k` tridiagonal matrix.

#### References

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
* H. I. van der Veen and K. Vuik, [*Bi-Lanczos with partial orthogonalization*](https://doi.org/10.1016/0045-7949(94)00565-K), Computers & structures, 56(4), pp. 605--613, 1995.
"""
function nonhermitian_lanczos(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
                              allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = γ₁ᴴ = zero(R)
  V = M(undef, n, k+1)
  U = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_Tᴴ)

  pαᵢ = 1  # Position of αᵢ and ᾱᵢ in the vectors `nzval_T` and `nzval_Tᴴ`
  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      cᴴb = kdot(n, c, b)
      if cᴴb == 0
        !allow_breakdown && error("Exact breakdown β₁γ₁ == 0.")
        βᵢ₊₁ = zero(FC)
        γᵢ₊₁ = zero(FC)
        kfill!(vᵢ₊₁, zero(FC))
        kfill!(uᵢ₊₁, zero(FC))
      else
        β₁ = √(abs(cᴴb))
        γ₁ᴴ = conj(cᴴb / β₁)
        kdivcopy!(n, vᵢ, b, β₁)
        kdivcopy!(n, uᵢ, c, γ₁ᴴ)
      end
    end
    kmul!(q, A , vᵢ)
    kmul!(p, Aᴴ, uᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      kaxpy!(n, -     γᵢ , vᵢ₋₁, q)
      kaxpy!(n, -conj(βᵢ), uᵢ₋₁, p)
    end
    αᵢ = kdot(n, uᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    kaxpy!(n, -     αᵢ , vᵢ, q)
    kaxpy!(n, -conj(αᵢ), uᵢ, p)
    pᴴq = kdot(n, p, q)
    if pᴴq == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁γᵢ₊₁ == 0 at iteration i = $i.")
      βᵢ₊₁ = zero(FC)
      γᵢ₊₁ = zero(FC)
      kfill!(vᵢ₊₁, zero(FC))
      kfill!(uᵢ₊₁, zero(FC))
    else
      βᵢ₊₁ = √(abs(pᴴq))
      γᵢ₊₁ = pᴴq / βᵢ₊₁
      kdivcopy!(n, vᵢ₊₁, q, βᵢ₊₁)
      kdivcopy!(n, uᵢ₊₁, p, conj(γᵢ₊₁))
    end
    nzval_T[pαᵢ+1]  = βᵢ₊₁        # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = conj(γᵢ₊₁)  # Tᴴᵢ₊₁.ᵢ = γ̄ᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁        # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = conj(βᵢ₊₁)  # Tᴴᵢ.ᵢ₊₁ = β̄ᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T, U, γ₁ᴴ, Tᴴ
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
        kdivcopy!(n, vⱼ, b, β)
      end
    end
    kmul!(q, A, vⱼ)
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
      kdivcopy!(n, vⱼ₊₁, q, H[j+1,j])
    end
  end
  return V, β, H
end

"""
    V, U, β, L = golub_kahan(A, b, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `k`: the number of iterations of the Golub-Kahan process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `n × (k+1)` matrix;
* `U`: a dense `m × (k+1)` matrix;
* `β`: a coefficient such that `βu₁ = b`;
* `L`: a sparse `(k+1) × (k+1)` lower bidiagonal matrix.

#### References

* G. H. Golub and W. Kahan, [*Calculating the Singular Values and Pseudo-Inverse of a Matrix*](https://doi.org/10.1137/0702016), SIAM Journal on Numerical Analysis, 2(2), pp. 225--224, 1965.
* C. C. Paige, [*Bidiagonalization of Matrices and Solution of Linear Equations*](https://doi.org/10.1137/0711019), SIAM Journal on Numerical Analysis, 11(1), pp. 197--209, 1974.
"""
function golub_kahan(A, b::AbstractVector{FC}, k::Int;
                     allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  Aᴴ = A'
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+2)
  rowval = zeros(Int, 2k+1)
  nzval = zeros(R, 2k+1)

  colptr[1] = 1
  for i = 1:k+1
    pos = colptr[i]
    if i ≤ k
      colptr[i+1] = pos + 2
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      colptr[i+1] = pos + 1
      rowval[pos] = i
    end
  end

  β₁ = zero(R)
  V = M(undef, n, k+1)
  U = M(undef, m, k+1)
  L = SparseMatrixCSC(k+1, k+1, colptr, rowval, nzval)

  pαᵢ = 1  # Position of αᵢ in the vector `nzval`
  for i = 1:k
    uᵢ = view(U,:,i)
    vᵢ = view(V,:,i)
    uᵢ₊₁ = q = view(U,:,i+1)
    vᵢ₊₁ = p = view(V,:,i+1)
    if i == 1
      wᵢ = vᵢ
      β₁ = knorm(m, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(uᵢ, zero(FC))
      else
        kdivcopy!(m, uᵢ, b, β₁)
      end
      kmul!(wᵢ, Aᴴ, uᵢ)
      αᵢ = knorm(n, wᵢ)
      if αᵢ == 0
        !allow_breakdown && error("Exact breakdown α₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        kdivcopy!(n, vᵢ, wᵢ, αᵢ)
      end
      nzval[pαᵢ] = αᵢ  # Lᵢ.ᵢ = αᵢ
    end
    kmul!(q, A, vᵢ)
    αᵢ = nzval[pαᵢ]  # αᵢ = Lᵢ.ᵢ
    kaxpy!(m, -αᵢ, uᵢ, q)
    βᵢ₊₁ = knorm(m, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      kdivcopy!(m, uᵢ₊₁, q, βᵢ₊₁)
    end
    kmul!(p, Aᴴ, uᵢ₊₁)
    kaxpy!(n, -βᵢ₊₁, vᵢ, p)
    αᵢ₊₁ = knorm(n, p)
    if αᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown αᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      kdivcopy!(n, vᵢ₊₁, p, αᵢ₊₁)
    end
    nzval[pαᵢ+1] = βᵢ₊₁  # Lᵢ₊₁.ᵢ   = βᵢ₊₁
    nzval[pαᵢ+2] = αᵢ₊₁  # Lᵢ₊₁.ᵢ₊₁ = αᵢ₊₁
    pαᵢ = pαᵢ + 2
  end
  return V, U, β₁, L
end

"""
    V, β, T, U, γᴴ, Tᴴ = saunders_simon_yip(A, b, c, k; allow_breakdown=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`;
* `k`: the number of iterations of the Saunders-Simon-Yip process.

#### Keyword argument

* `allow_breakdown`: specify whether to continue the process or raise an error when an exact breakdown occurs.

#### Output arguments

* `V`: a dense `m × (k+1)` matrix;
* `β`: a coefficient such that `βv₁ = b`;
* `T`: a sparse `(k+1) × k` tridiagonal matrix;
* `U`: a dense `n × (k+1)` matrix;
* `γᴴ`: a coefficient such that `γᴴu₁ = c`;
* `Tᴴ`: a sparse `(k+1) × k` tridiagonal matrix.

#### Reference

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
"""
function saunders_simon_yip(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
                            allow_breakdown::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = 3i
    if i == 1
      rowval[pos] = i
      rowval[pos+1] = i+1
    else
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  β₁ = γ₁ᴴ = zero(R)
  V = M(undef, m, k+1)
  U = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_Tᴴ)

  pαᵢ = 1  # Position of αᵢ and ᾱᵢ in the vectors `nzval_T` and `nzval_Tᴴ`
  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      β₁ = knorm(m, b)
      if β₁ == 0
        !allow_breakdown && error("Exact breakdown β₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        kdivcopy!(m, vᵢ, b, β₁)
      end
      γ₁ᴴ = knorm(n, c)
      if γ₁ᴴ == 0
        !allow_breakdown && error("Exact breakdown γ₁ᴴ == 0.")
        kfill!(uᵢ, zero(FC))
      else
        kdivcopy!(n, uᵢ, c, γ₁ᴴ)
      end
    end
    kmul!(q, A , uᵢ)
    kmul!(p, Aᴴ, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      kaxpy!(m, -γᵢ, vᵢ₋₁, q)
      kaxpy!(n, -βᵢ, uᵢ₋₁, p)
    end
    αᵢ = kdot(m, vᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    kaxpy!(m, -     αᵢ , vᵢ, q)
    kaxpy!(n, -conj(αᵢ), uᵢ, p)
    βᵢ₊₁ = knorm(m, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      kdivcopy!(m, vᵢ₊₁, q, βᵢ₊₁)
    end
    γᵢ₊₁ = knorm(n, p)
    if γᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown γᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      kdivcopy!(n, uᵢ₊₁, p, γᵢ₊₁)
    end
    nzval_T[pαᵢ+1]  = βᵢ₊₁  # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = γᵢ₊₁  # Tᴴᵢ₊₁.ᵢ = γᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁  # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = βᵢ₊₁  # Tᴴᵢ.ᵢ₊₁ = βᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, β₁, T, U, γ₁ᴴ, Tᴴ
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
        kdivcopy!(m, vⱼ, b, β)
      end
      γ = knorm(n, c)
      if γ == 0
        !allow_breakdown && error("Exact breakdown γ == 0.")
        kfill!(uⱼ, zero(FC))
      else
        kdivcopy!(n, uⱼ, c, γ)
      end
    end
    kmul!(q, A, uⱼ)
    kmul!(p, B, vⱼ)
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
      kdivcopy!(m, vⱼ₊₁, q, H[j+1,j])
    end
    F[j+1,j] = knorm(n, p)
    if F[j+1,j] == 0
      !allow_breakdown && error("Exact breakdown Fᵢ₊₁.ᵢ == 0 at iteration i = $j.")
      kfill!(uⱼ₊₁, zero(FC))
    else
      kdivcopy!(n, uⱼ₊₁, p, F[j+1,j])
    end
  end
  return V, β, H, U, γ, F
end
