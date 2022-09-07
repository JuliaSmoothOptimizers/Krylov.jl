export symmetric_lanczos, unsymmetric_lanczos, arnoldi, golub_kahan, saunders_simon_yip, montoison_orban

"""
    V, T = symmetric_lanczos(A, b, k)

#### Input arguments:

* `A`: a linear operator that models an Hermitian matrix of dimension n.
* `b`: a vector of length n.
* `k`: the number of iterations of the symmetric Lanczos process.

#### Output arguments:

* `V`: a dense n × (k+1) matrix.
* `T`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function symmetric_lanczos(A, b::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)
  R = real(FC)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval = zeros(R, 3k-1)

  colptr[1] = 1
  rowval[1] = 1
  rowval[2] = 2
  for i = 1:k
    colptr[i+1] = 3i
    if i ≥ 2
      pos = colptr[i]
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval)
  V = zeros(FC, n, k+1)

  for i = 1:k
    vᵢ = view(V,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    if i == 1
      βᵢ = @knrm2(n, b)
      vᵢ .= b ./ βᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = @kdotr(n, vᵢ, q)
    T[i,i] = αᵢ
    @kaxpy!(n, -αᵢ, vᵢ, q)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      βᵢ = T[i,i-1]
      T[i-1,i] = βᵢ
      @kaxpy!(n, -βᵢ, vᵢ₋₁, q)
    end
    βᵢ₊₁ = @knrm2(n, q)
    T[i+1,i] = βᵢ₊₁
    vᵢ₊₁ .= q ./ βᵢ₊₁
  end
  return V, T
end

"""
    V, T, U, S = unsymmetric_lanczos(A, b, c, k)

#### Input arguments:

* `A`: a linear operator that models a square matrix of dimension n.
* `b`: a vector of length n.
* `c`: a vector of length n.
* `k`: the number of iterations of the unsymmetric Lanczos process.

#### Output arguments:

* `V`: a dense n × (k+1) matrix.
* `T`: a sparse (k+1) × k tridiagonal matrix.
* `U`: a dense n × (k+1) matrix.
* `S`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function unsymmetric_lanczos(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)
  Aᴴ = A'

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_S = zeros(FC, 3k-1)

  colptr[1] = 1
  rowval[1] = 1
  rowval[2] = 2
  for i = 1:k
    colptr[i+1] = 3i
    if i ≥ 2
      pos = colptr[i]
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  S = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_S)
  V = zeros(FC, n, k+1)
  U = zeros(FC, n, k+1)

  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      cᴴb = @kdot(n, c, b)
      βᵢ = √(abs(cᴴb))
      γᵢ = cᴴb / βᵢ
      vᵢ .= b ./ βᵢ
      uᵢ .= c ./ conj(γᵢ)
    end
    mul!(q, A , vᵢ)
    mul!(p, Aᴴ, uᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = T[i,i-1]
      γᵢ = T[i-1,i]
      @kaxpy!(n, -     γᵢ , vᵢ₋₁, q)
      @kaxpy!(n, -conj(βᵢ), uᵢ₋₁, p)
    end
    αᵢ = @kdot(n, uᵢ, q)
    T[i,i] = αᵢ
    S[i,i] = conj(αᵢ)
    @kaxpy!(m, -     αᵢ , vᵢ, q)
    @kaxpy!(n, -conj(αᵢ), uᵢ, p)
    pᴴq = @kdot(n, p, q)
    βᵢ₊₁ = √(abs(pᴴq))
    γᵢ₊₁ = pᴴq / βᵢ₊₁
    vᵢ₊₁ .= q ./ βᵢ₊₁
    uᵢ₊₁ .= p ./ conj(γᵢ₊₁)
    T[i+1,i] = βᵢ₊₁
    S[i+1,i] = conj(γᵢ₊₁)
    if i ≤ k-1
      T[i,i+1] = γᵢ₊₁
      S[i,i+1] = conj(βᵢ₊₁)
    end
  end
  return V, T, U, S
end

"""
    V, H = arnoldi(A, b, k)

#### Input arguments:

* `A`: a linear operator that models a square matrix of dimension n.
* `b`: a vector of length n.
* `k`: the number of iterations of the Arnoldi process.

#### Output arguments:

* `V`: a dense n × (k+1) matrix.
* `H`: a sparse (k+1) × k upper Hessenberg matrix.

#### Reference

* W. E. Arnoldi, [*The principle of minimized iterations in the solution of the matrix eigenvalue problem*](https://doi.org/10.1090/qam/42792), Quarterly of Applied Mathematics, 9, pp. 17--29, 1951.
"""
function arnoldi(A, b::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)

  nnz = div(k*(k+1), 2) + k
  colptr = zeros(Int, k+1)
  rowval = zeros(Int, nnz)
  nzval = zeros(FC, nnz)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = pos+i+1
    for j = 1:i+1
      rowval[pos+j-1] = j
    end
  end

  H = SparseMatrixCSC(k+1, k, colptr, rowval, nzval)
  V = zeros(FC, n, k+1)

  for i = 1:k
    vᵢ = view(V,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    if i == 1
      β = @knrm2(n, b)
      vᵢ .= b ./ β
    end
    mul!(q, A, vᵢ)
    for j = 1:i
      vⱼ = view(V,:,j)
      H[j,i] = @kdot(n, vⱼ, q)
      @kaxpy!(n, -H[j,i], vⱼ, q)
    end
    H[i+1,i] = @knrm2(n, q)
    vᵢ₊₁ .= q ./ H[i+1,i]
  end
  return V, H
end

"""
    V, U, L = golub_kahan(A, b, k)

#### Input arguments:

* `A`: a linear operator that models a rectangular matrix of dimension n × m.
* `b`: a vector of length n.
* `k`: the number of iterations of the Golub-Kahan process.

#### Output arguments:

* `V`: a dense m × (k+1) matrix.
* `U`: a dense n × (k+1) matrix.
* `L`: a sparse (k+1) × (k+1) lower bidiagonal matrix.

#### Reference

* G. H. Golub and W. Kahan, [*Calculating the Singular Values and Pseudo-Inverse of a Matrix*](https://doi.org/10.1137/0702016), SIAM Journal on Numerical Analysis, 2(2), pp. 225--224, 1965.
"""
function golub_kahan(A, b::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)
  R = real(FC)
  Aᴴ = A'

  colptr = zeros(Int, k+2)
  rowval = zeros(Int, 2k+1)
  nzval = zeros(R, 2k+1)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = pos+2
    rowval[pos] = i
    rowval[pos+1] = i+1
  end
  rowval[2k+1] = k+1
  colptr[k+2] = 2k+2

  L = SparseMatrixCSC(k+1, k+1, colptr, rowval, nzval)
  V = zeros(FC, m, k+1)
  U = zeros(FC, n, k+1)

  for i = 1:k
    uᵢ = view(U,:,i)
    vᵢ = view(V,:,i)
    uᵢ₊₁ = q = view(U,:,i+1)
    vᵢ₊₁ = p = view(V,:,i+1)
    if i == 1
      wᵢ = vᵢ
      βᵢ = @knrm2(n, b)
      uᵢ .= b ./ βᵢ
      mul!(wᵢ, Aᴴ, uᵢ)
      αᵢ = @knrm2(m, wᵢ)
      L[1,1] = αᵢ
      vᵢ .= wᵢ ./ αᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = L[i,i] 
    @kaxpy!(n, -αᵢ, uᵢ, q)
    βᵢ₊₁ = @knrm2(n, q)
    uᵢ₊₁ .= q ./ βᵢ₊₁
    mul!(p, Aᴴ, uᵢ₊₁)
    @kaxpy!(m, -βᵢ₊₁, vᵢ, p)
    αᵢ₊₁ = @knrm2(m, p)
    vᵢ₊₁ .= p ./ αᵢ₊₁
    L[i+1,i]   = βᵢ₊₁
    L[i+1,i+1] = αᵢ₊₁
  end
  return V, U, L
end

"""
    V, T, U, S = saunders_simon_yip(A, b, c, k)

#### Input arguments:

* `A`: a linear operator that models a rectangular matrix of dimension n × m.
* `b`: a vector of length n.
* `c`: a vector of length m.
* `k`: the number of iterations of the Saunders-Simon-Yip process.

#### Output arguments:

* `V`: a dense n × (k+1) matrix.
* `T`: a sparse (k+1) × k tridiagonal matrix.
* `U`: a dense m × (k+1) matrix.
* `S`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
"""
function saunders_simon_yip(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)
  Aᴴ = A'

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_S = zeros(FC, 3k-1)

  colptr[1] = 1
  rowval[1] = 1
  rowval[2] = 2
  for i = 1:k
    colptr[i+1] = 3i
    if i ≥ 2
      pos = colptr[i]
      rowval[pos] = i-1
      rowval[pos+1] = i
      rowval[pos+2] = i+1
    end
  end

  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_T)
  S = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_S)
  V = zeros(FC, n, k+1)
  U = zeros(FC, m, k+1)

  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      β = @knrm2(n, b)
      γ = @knrm2(m, c)
      vᵢ .= b ./ β
      uᵢ .= c ./ γ
    end
    mul!(q, A , uᵢ)
    mul!(p, Aᴴ, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = T[i,i-1]
      γᵢ = T[i-1,i]
      @kaxpy!(n, -γᵢ, vᵢ₋₁, q)
      @kaxpy!(m, -βᵢ, uᵢ₋₁, p)
    end
    αᵢ = @kdot(n, vᵢ, q)
    T[i,i] = αᵢ
    S[i,i] = conj(αᵢ)
    @kaxpy!(n, -     αᵢ , vᵢ, q)
    @kaxpy!(m, -conj(αᵢ), uᵢ, p)
    βᵢ₊₁ = @knrm2(n, q)
    γᵢ₊₁ = @knrm2(m, p)
    vᵢ₊₁ .= q ./ βᵢ₊₁
    uᵢ₊₁ .= p ./ γᵢ₊₁
    T[i+1,i] = βᵢ₊₁
    S[i+1,i] = conj(γᵢ₊₁)
    if i ≤ k-1
      T[i,i+1] = γᵢ₊₁
      S[i,i+1] = conj(βᵢ₊₁)
    end
  end
  return V, T, U, S
end

"""
    V, H, U, F = montoison_orban(A, B, b, c, k)

#### Input arguments:

* `A`: a linear operator that models a rectangular matrix of dimension n × m.
* `B`: a linear operator that models a rectangular matrix of dimension m × n.
* `b`: a vector of length n.
* `c`: a vector of length m.
* `k`: the number of iterations of the Montoison-Orban process.

#### Output arguments:

* `V`: a dense n × (k+1) matrix.
* `H`: a sparse (k+1) × k upper Hessenberg matrix.
* `U`: a dense m × (k+1) matrix.
* `F`: a sparse (k+1) × k upper Hessenberg matrix.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://dx.doi.org/10.13140/RG.2.2.24069.68326), Cahier du GERAD G-2021-62, GERAD, Montréal, 2021.
"""
function montoison_orban(A, B, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  n, m = size(A)
  
  nnz = div(k*(k+1), 2) + k
  colptr = zeros(Int, k+1)
  rowval = zeros(Int, nnz)
  nzval_H = zeros(FC, nnz)
  nzval_F = zeros(FC, nnz)

  colptr[1] = 1
  for i = 1:k
    pos = colptr[i]
    colptr[i+1] = pos+i+1
    for j = 1:i+1
      rowval[pos+j-1] = j
    end
  end

  H = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_H)
  F = SparseMatrixCSC(k+1, k, colptr, rowval, nzval_F)
  V = zeros(FC, n, k+1)
  U = zeros(FC, m, k+1)

  for i = 1:k
    vᵢ = view(V,:,i)
    uᵢ = view(U,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    uᵢ₊₁ = p = view(U,:,i+1)
    if i == 1
      β = @knrm2(n, b)
      γ = @knrm2(m, c)
      vᵢ .= b ./ β
      uᵢ .= c ./ γ
    end
    mul!(q, A, uᵢ)
    mul!(p, B, vᵢ)
    for j = 1:i
      vⱼ = view(V,:,j)
      uⱼ = view(U,:,j)
      H[j,i] = @kdot(n, vⱼ, q)
      @kaxpy!(n, -H[j,i], vⱼ, q)
      F[j,i] = @kdot(m, uⱼ, p)
      @kaxpy!(m, -F[j,i], uⱼ, p)
    end
    H[i+1,i] = @knrm2(n, q)
    vᵢ₊₁ .= q ./ H[i+1,i]
    F[i+1,i] = @knrm2(m, p)
    uᵢ₊₁ .= p ./ F[i+1,i]
  end
  return V, H, U, F
end
