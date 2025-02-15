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
function Krylov.hermitian_lanczos(A, b::AbstractVector{FC}, k::Int;
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
        vᵢ .= b ./ β₁
      end
    end
    mul!(q, A, vᵢ)
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
      vᵢ₊₁ .= q ./ βᵢ₊₁
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
function Krylov.nonhermitian_lanczos(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
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
        vᵢ .= b ./ β₁
        uᵢ .= c ./ γ₁ᴴ
      end
    end
    mul!(q, A , vᵢ)
    mul!(p, Aᴴ, uᵢ)
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
      vᵢ₊₁ .= q ./ βᵢ₊₁
      uᵢ₊₁ .= p ./ conj(γᵢ₊₁)
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
function Krylov.golub_kahan(A, b::AbstractVector{FC}, k::Int;
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
        uᵢ .= b ./ β₁
      end
      mul!(wᵢ, Aᴴ, uᵢ)
      αᵢ = knorm(n, wᵢ)
      if αᵢ == 0
        !allow_breakdown && error("Exact breakdown α₁ == 0.")
        kfill!(vᵢ, zero(FC))
      else
        vᵢ .= wᵢ ./ αᵢ
      end
      nzval[pαᵢ] = αᵢ  # Lᵢ.ᵢ = αᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = nzval[pαᵢ]  # αᵢ = Lᵢ.ᵢ
    kaxpy!(m, -αᵢ, uᵢ, q)
    βᵢ₊₁ = knorm(m, q)
    if βᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown βᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      uᵢ₊₁ .= q ./ βᵢ₊₁
    end
    mul!(p, Aᴴ, uᵢ₊₁)
    kaxpy!(n, -βᵢ₊₁, vᵢ, p)
    αᵢ₊₁ = knorm(n, p)
    if αᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown αᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(vᵢ₊₁, zero(FC))
    else
      vᵢ₊₁ .= p ./ αᵢ₊₁
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
function Krylov.saunders_simon_yip(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int;
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
        vᵢ .= b ./ β₁
      end
      γ₁ᴴ = knorm(n, c)
      if γ₁ᴴ == 0
        !allow_breakdown && error("Exact breakdown γ₁ᴴ == 0.")
        kfill!(uᵢ, zero(FC))
      else
        uᵢ .= c ./ γ₁ᴴ
      end
    end
    mul!(q, A , uᵢ)
    mul!(p, Aᴴ, vᵢ)
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
      vᵢ₊₁ .= q ./ βᵢ₊₁
    end
    γᵢ₊₁ = knorm(n, p)
    if γᵢ₊₁ == 0
      !allow_breakdown && error("Exact breakdown γᵢ₊₁ == 0 at iteration i = $i.")
      kfill!(uᵢ₊₁, zero(FC))
    else
      uᵢ₊₁ .= p ./ γᵢ₊₁
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
