export hermitian_lanczos, nonhermitian_lanczos, arnoldi, golub_kahan, saunders_simon_yip, montoison_orban

"""
    V, T = hermitian_lanczos(A, b, k)

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension n;
* `b`: a vector of length n;
* `k`: the number of iterations of the Hermitian Lanczos process.

#### Output arguments

* `V`: a dense n × (k+1) matrix;
* `T`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function hermitian_lanczos(A, b::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  S = ktypeof(b)
  M = vector_to_matrix(S)

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

  V = M(undef, n, k+1)
  T = SparseMatrixCSC(k+1, k, colptr, rowval, nzval)

  pαᵢ = 1  # Position of αᵢ in the vector `nzval`
  for i = 1:k
    vᵢ = view(V,:,i)
    vᵢ₊₁ = q = view(V,:,i+1)
    if i == 1
      βᵢ = @knrm2(n, b)
      vᵢ .= b ./ βᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = @kdotr(n, vᵢ, q)
    nzval[pαᵢ] = αᵢ  # Tᵢ.ᵢ = αᵢ
    @kaxpy!(n, -αᵢ, vᵢ, q)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      βᵢ = nzval[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      nzval[pαᵢ-1] = βᵢ  # Tᵢ₋₁.ᵢ = βᵢ
      @kaxpy!(n, -βᵢ, vᵢ₋₁, q)
    end
    βᵢ₊₁ = @knrm2(n, q)
    nzval[pαᵢ+1] = βᵢ₊₁  # Tᵢ₊₁.ᵢ = βᵢ₊₁
    vᵢ₊₁ .= q ./ βᵢ₊₁
    pαᵢ = pαᵢ + 3
  end
  return V, T
end

"""
    V, T, U, Tᴴ = nonhermitian_lanczos(A, b, c, k)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension n;
* `b`: a vector of length n;
* `c`: a vector of length n;
* `k`: the number of iterations of the non-Hermitian Lanczos process.

#### Output arguments

* `V`: a dense n × (k+1) matrix;
* `T`: a sparse (k+1) × k tridiagonal matrix;
* `U`: a dense n × (k+1) matrix;
* `Tᴴ`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* C. Lanczos, [*An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*](https://doi.org/10.6028/jres.045.026), Journal of Research of the National Bureau of Standards, 45(4), pp. 225--280, 1950.
"""
function nonhermitian_lanczos(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

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
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      @kaxpy!(n, -     γᵢ , vᵢ₋₁, q)
      @kaxpy!(n, -conj(βᵢ), uᵢ₋₁, p)
    end
    αᵢ = @kdot(n, uᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    @kaxpy!(m, -     αᵢ , vᵢ, q)
    @kaxpy!(n, -conj(αᵢ), uᵢ, p)
    pᴴq = @kdot(n, p, q)
    βᵢ₊₁ = √(abs(pᴴq))
    γᵢ₊₁ = pᴴq / βᵢ₊₁
    vᵢ₊₁ .= q ./ βᵢ₊₁
    uᵢ₊₁ .= p ./ conj(γᵢ₊₁)
    nzval_T[pαᵢ+1]  = βᵢ₊₁        # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = conj(γᵢ₊₁)  # Tᴴᵢ₊₁.ᵢ = γ̄ᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁        # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = conj(βᵢ₊₁)  # Tᴴᵢ.ᵢ₊₁ = β̄ᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, T, U, Tᴴ
end

"""
    V, H = arnoldi(A, b, k; reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension n;
* `b`: a vector of length n;
* `k`: the number of iterations of the Arnoldi process.

#### Keyword arguments

* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against all previous vectors.

#### Output arguments

* `V`: a dense n × (k+1) matrix;
* `H`: a dense (k+1) × k upper Hessenberg matrix.

#### Reference

* W. E. Arnoldi, [*The principle of minimized iterations in the solution of the matrix eigenvalue problem*](https://doi.org/10.1090/qam/42792), Quarterly of Applied Mathematics, 9, pp. 17--29, 1951.
"""
function arnoldi(A, b::AbstractVector{FC}, k::Int; reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  S = ktypeof(b)
  M = vector_to_matrix(S)

  V = M(undef, n, k+1)
  H = zeros(FC, k+1, k)

  for j = 1:k
    vⱼ = view(V,:,j)
    vⱼ₊₁ = q = view(V,:,j+1)
    if j == 1
      β = @knrm2(n, b)
      vⱼ .= b ./ β
    end
    mul!(q, A, vⱼ)
    for i = 1:j
      vᵢ = view(V,:,i)
      H[i,j] = @kdot(n, vᵢ, q)
      @kaxpy!(n, -H[i,j], vᵢ, q)
    end
    if reorthogonalization
      for i = 1:j
        vᵢ = view(V,:,i)
        Htmp = @kdot(n, vᵢ, q)
        @kaxpy!(n, -Htmp, vᵢ, q)
        H[i,j] += Htmp
      end
    end
    H[j+1,j] = @knrm2(n, q)
    vⱼ₊₁ .= q ./ H[j+1,j]
  end
  return V, H
end

"""
    V, U, L = golub_kahan(A, b, k)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m;
* `k`: the number of iterations of the Golub-Kahan process.

#### Output arguments

* `V`: a dense n × (k+1) matrix;
* `U`: a dense m × (k+1) matrix;
* `L`: a sparse (k+1) × (k+1) lower bidiagonal matrix.

#### References

* G. H. Golub and W. Kahan, [*Calculating the Singular Values and Pseudo-Inverse of a Matrix*](https://doi.org/10.1137/0702016), SIAM Journal on Numerical Analysis, 2(2), pp. 225--224, 1965.
* C. C. Paige, [*Bidiagonalization of Matrices and Solution of Linear Equations*](https://doi.org/10.1137/0711019), SIAM Journal on Numerical Analysis, 11(1), pp. 197--209, 1974.
"""
function golub_kahan(A, b::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  R = real(FC)
  Aᴴ = A'
  S = ktypeof(b)
  M = vector_to_matrix(S)

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
      βᵢ = @knrm2(m, b)
      uᵢ .= b ./ βᵢ
      mul!(wᵢ, Aᴴ, uᵢ)
      αᵢ = @knrm2(n, wᵢ)
      nzval[pαᵢ] = αᵢ  # Lᵢ.ᵢ = αᵢ
      vᵢ .= wᵢ ./ αᵢ
    end
    mul!(q, A, vᵢ)
    αᵢ = nzval[pαᵢ]  # αᵢ = Lᵢ.ᵢ
    @kaxpy!(m, -αᵢ, uᵢ, q)
    βᵢ₊₁ = @knrm2(m, q)
    uᵢ₊₁ .= q ./ βᵢ₊₁
    mul!(p, Aᴴ, uᵢ₊₁)
    @kaxpy!(n, -βᵢ₊₁, vᵢ, p)
    αᵢ₊₁ = @knrm2(n, p)
    vᵢ₊₁ .= p ./ αᵢ₊₁
    nzval[pαᵢ+1] = βᵢ₊₁  # Lᵢ₊₁.ᵢ   = βᵢ₊₁
    nzval[pαᵢ+2] = αᵢ₊₁  # Lᵢ₊₁.ᵢ₊₁ = αᵢ₊₁
    pαᵢ = pαᵢ + 2
  end
  return V, U, L
end

"""
    V, T, U, Tᴴ = saunders_simon_yip(A, b, c, k)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m;
* `c`: a vector of length n;
* `k`: the number of iterations of the Saunders-Simon-Yip process.

#### Output arguments

* `V`: a dense m × (k+1) matrix;
* `T`: a sparse (k+1) × k tridiagonal matrix;
* `U`: a dense n × (k+1) matrix;
* `Tᴴ`: a sparse (k+1) × k tridiagonal matrix.

#### Reference

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
"""
function saunders_simon_yip(A, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  Aᴴ = A'
  S = ktypeof(b)
  M = vector_to_matrix(S)

  colptr = zeros(Int, k+1)
  rowval = zeros(Int, 3k-1)
  nzval_T = zeros(FC, 3k-1)
  nzval_Tᴴ = zeros(FC, 3k-1)

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
      β = @knrm2(m, b)
      γ = @knrm2(n, c)
      vᵢ .= b ./ β
      uᵢ .= c ./ γ
    end
    mul!(q, A , uᵢ)
    mul!(p, Aᴴ, vᵢ)
    if i ≥ 2
      vᵢ₋₁ = view(V,:,i-1)
      uᵢ₋₁ = view(U,:,i-1)
      βᵢ = nzval_T[pαᵢ-2]  # βᵢ = Tᵢ.ᵢ₋₁
      γᵢ = nzval_T[pαᵢ-1]  # γᵢ = Tᵢ₋₁.ᵢ
      @kaxpy!(m, -γᵢ, vᵢ₋₁, q)
      @kaxpy!(n, -βᵢ, uᵢ₋₁, p)
    end
    αᵢ = @kdot(m, vᵢ, q)
    nzval_T[pαᵢ]  = αᵢ        # Tᵢ.ᵢ  = αᵢ
    nzval_Tᴴ[pαᵢ] = conj(αᵢ)  # Tᴴᵢ.ᵢ = ᾱᵢ
    @kaxpy!(m, -     αᵢ , vᵢ, q)
    @kaxpy!(n, -conj(αᵢ), uᵢ, p)
    βᵢ₊₁ = @knrm2(m, q)
    γᵢ₊₁ = @knrm2(n, p)
    vᵢ₊₁ .= q ./ βᵢ₊₁
    uᵢ₊₁ .= p ./ γᵢ₊₁
    nzval_T[pαᵢ+1]  = βᵢ₊₁  # Tᵢ₊₁.ᵢ  = βᵢ₊₁
    nzval_Tᴴ[pαᵢ+1] = γᵢ₊₁  # Tᴴᵢ₊₁.ᵢ = γᵢ₊₁
    if i ≤ k-1
      nzval_T[pαᵢ+2]  = γᵢ₊₁  # Tᵢ.ᵢ₊₁  = γᵢ₊₁
      nzval_Tᴴ[pαᵢ+2] = βᵢ₊₁  # Tᴴᵢ.ᵢ₊₁ = βᵢ₊₁
    end
    pαᵢ = pαᵢ + 3
  end
  return V, T, U, Tᴴ
end

"""
    V, H, U, F = montoison_orban(A, B, b, c, k; reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `B`: a linear operator that models a matrix of dimension n × m;
* `b`: a vector of length m;
* `c`: a vector of length n;
* `k`: the number of iterations of the Montoison-Orban process.

#### Keyword arguments

* `reorthogonalization`: reorthogonalize the new vectors of the Krylov basis against all previous vectors.

#### Output arguments

* `V`: a dense m × (k+1) matrix;
* `H`: a dense (k+1) × k upper Hessenberg matrix;
* `U`: a dense n × (k+1) matrix;
* `F`: a dense (k+1) × k upper Hessenberg matrix.

#### Reference

* A. Montoison and D. Orban, [*GPMR: An Iterative Method for Unsymmetric Partitioned Linear Systems*](https://doi.org/10.1137/21M1459265), SIAM Journal on Matrix Analysis and Applications, 44(1), pp. 293--311, 2023.
"""
function montoison_orban(A, B, b::AbstractVector{FC}, c::AbstractVector{FC}, k::Int; reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  S = ktypeof(b)
  M = vector_to_matrix(S)

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
      β = @knrm2(m, b)
      γ = @knrm2(n, c)
      vⱼ .= b ./ β
      uⱼ .= c ./ γ
    end
    mul!(q, A, uⱼ)
    mul!(p, B, vⱼ)
    for i = 1:j
      vᵢ = view(V,:,i)
      uᵢ = view(U,:,i)
      H[i,j] = @kdot(m, vᵢ, q)
      @kaxpy!(n, -H[i,j], vᵢ, q)
      F[i,j] = @kdot(n, uᵢ, p)
      @kaxpy!(m, -F[i,j], uᵢ, p)
    end
    if reorthogonalization
      for i = 1:j
        vᵢ = view(V,:,i)
        uᵢ = view(U,:,i)
        Htmp = @kdot(m, vᵢ, q)
        @kaxpy!(m, -Htmp, vᵢ, q)
        H[i,j] += Htmp
        Ftmp = @kdot(n, uᵢ, p)
        @kaxpy!(n, -Ftmp, uᵢ, p)
        F[i,j] += Ftmp
      end
    end
    H[j+1,j] = @knrm2(m, q)
    vⱼ₊₁ .= q ./ H[j+1,j]
    F[j+1,j] = @knrm2(n, p)
    uⱼ₊₁ .= p ./ F[j+1,j]
  end
  return V, H, U, F
end
