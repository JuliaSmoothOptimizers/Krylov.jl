"""
    V, Ψ, T = hermitian_lanczos(A, B, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a Hermitian matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Hermitian Lanczos process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function hermitian_lanczos(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval)

  α = -one(FC)
  β = one(FC)
  q = zeros(FC, n, p)
  ψ₁ = zeros(FC, p, p)
  Ωᵢ = Ψᵢ = Ψᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)

    if i == 1
      q .= B
      reduced_qr!(q, ψ₁, algo)
      vᵢ .= q
    end

    mul!(q, A, vᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      mul!(q, vᵢ₋₁, Ψᵢ', α, β)  # q = q - vᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, vᵢ', q)       # Ωᵢ = vᵢᴴ * q
    mul!(q, vᵢ, Ωᵢ, α, β)  # q = q - vᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
      end
    end

    reduced_qr!(q, Ψᵢ₊₁, algo)
    vᵢ₊₁ .= q

    # Store the block Ψᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (T[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, ψ₁, T
end

"""
    V, Ψ, T, U, Φᴴ, Tᴴ = nonhermitian_lanczos(A, B, C, k)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block non-Hermitian Lanczos process.

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Φᴴ`: a dense `p × p` upper triangular matrix such that `U₁Φᴴ = C`;
* `Tᴴ`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function nonhermitian_lanczos(A, B::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'
  pivoting = VERSION < v"1.9" ? Val{false}() : NoPivot()

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval_T = zeros(FC, nnzT)
  nzval_Tᴴ = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_Tᴴ)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, n, p)
  qᵤ = zeros(FC, n, p)
  D = Ωᵢ = zeros(FC, p, p)
  Ψ₁ = zeros(FC, p, p)
  Φ₁ᴴ = zeros(FC, p, p)

  local Φᵢ, Ψᵢ

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      mul!(D, C', B)  # D = Cᴴ * B
      F = lu(D, pivoting)
      Φᵢ, Ψᵢ = F.L, F.U   # Φᵢ = F.P' * Φᵢ with pivoting
      Ψ₁ .= Ψᵢ
      Φ₁ᴴ .= Φᵢ'
      # vᵢ .= (Ψᵢ' \ B')'
      # uᵢ .= (Φᵢ \ C')'
      ldiv!(vᵢ', UpperTriangular(Ψᵢ)', B')
      ldiv!(uᵢ', LowerTriangular(Φᵢ), C')
    end

    mul!(qᵥ, A, vᵢ)
    mul!(qᵤ, Aᴴ, uᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      uᵢ₋₁ = view(U,:,pos5:pos6)

      mul!(qᵥ, vᵢ₋₁, Φᵢ, α, β)   # qᵥ = qᵥ - vᵢ₋₁ * Φᵢ
      mul!(qᵤ, uᵢ₋₁, Ψᵢ', α, β)  # qᵤ = qᵤ - uᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, uᵢ', qᵥ)
    mul!(qᵥ, vᵢ, Ωᵢ, α, β)   # qᵥ = qᵥ - vᵢ * Ωᵢ
    mul!(qᵤ, uᵢ, Ωᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
        Tᴴ[indi,indj] = conj(Ωᵢ[jj,ii])
      end
    end

    mul!(D, qᵤ', qᵥ)  # D = qᵤᴴ * qᵥ
    F = lu(D, pivoting)
    Φᵢ₊₁, Ψᵢ₊₁ = F.L, F.U  # Φᵢ₊₁ = F.P' * Φᵢ₊₁ with pivoting
    # vᵢ₊₁ .= (Ψᵢ₊₁' \ qᵥ')'
    # uᵢ₊₁ .= (Φᵢ₊₁ \ qᵤ')'
    ldiv!(vᵢ₊₁', UpperTriangular(Ψᵢ₊₁)', qᵥ')
    ldiv!(uᵢ₊₁', LowerTriangular(Φᵢ₊₁), qᵤ')
    Φᵢ = Φᵢ₊₁
    Ψᵢ = Ψᵢ₊₁

    # Store the blocks Ψᵢ₊₁ and Φᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (Tᴴ[indi,indj] = conj(Φᵢ₊₁[jj,ii]))
        (ii ≤ jj) && (i < k) && (Tᴴ[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
        (ii ≤ jj) && (i < k) && (T[indj,indi] = Φᵢ₊₁[jj,ii])
      end
    end
  end
  return V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ
end

"""
    V, Γ, H = arnoldi(A, B, k; algo="householder", reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a square matrix of dimension `n`;
* `B`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Arnoldi process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).
* `reorthogonalization`: reorthogonalize the new matrices of the block Krylov basis against all previous matrices.

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `Γ`: a dense `p × p` upper triangular matrix such that `V₁Γ = B`;
* `H`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`.
"""
function arnoldi(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder", reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)

  V = zeros(FC, n, (k+1)*p)
  H = zeros(FC, (k+1)*p, k*p)

  α = -one(FC)
  β = one(FC)
  q = zeros(FC, n, p)
  Γ = zeros(FC, p, p)
  Ψᵢⱼ = Ψtmp = zeros(FC, p, p)

  for j = 1:k
    pos1 = (j-1)*p + 1
    pos2 = j*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vⱼ = view(V,:,pos1:pos2)
    vⱼ₊₁ = view(V,:,pos3:pos4)

    if j == 1
      q .= B
      reduced_qr!(q, Γ, algo)
      vⱼ .= q
    end

    mul!(q, A, vⱼ)

    for i = 1:j
      pos5 = (i-1)*p + 1
      pos6 = i*p
      vᵢ = view(V,:,pos5:pos6)
      mul!(Ψᵢⱼ, vᵢ', q)       # Ψᵢⱼ = vᵢᴴ * q
      mul!(q, vᵢ, Ψᵢⱼ, α, β)  # q = q - vᵢ * Ψᵢⱼ
      H[pos5:pos6,pos1:pos2] .= Ψᵢⱼ
    end

    if reorthogonalization
      for i = 1:j
        pos5 = (i-1)*p + 1
        pos6 = i*p
        vᵢ = view(V,:,pos5:pos6)
        mul!(Ψtmp, vᵢ', q)       # Ψtmp = vᵢᴴ * q
        mul!(q, vᵢ, Ψtmp, α, β)  # q = q - vᵢ * Ψtmp
        Hᵢⱼ = view(H,pos5:pos6,pos1:pos2)
        Hᵢⱼ .= Hᵢⱼ .+ Ψtmp
      end
    end

    Ψ = view(H, pos3:pos4,pos1:pos2)
    reduced_qr!(q, Ψ, algo)
    vⱼ₊₁ .= q
  end
  return V, Γ, H
end

"""
    V, U, Ψ, L = golub_kahan(A, B, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a matrix of size `m × p`;
* `k`: the number of iterations of the block Golub-Kahan process.

#### Keyword argument

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `n × p(k+1)` matrix;
* `U`: a dense `m × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `U₁Ψ = B`;
* `L`: a sparse `p(k+1) × p(k+1)` block lower bidiagonal matrix with a lower bandwidth `p`.
"""
function golub_kahan(A, B::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'

  nnzL = p*k*(p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*(k+1)+1)
  rowval = zeros(Int, nnzL)
  nzval = zeros(FC, nnzL)

  colptr[1] = 1
  for j = 1:(k+1)*p
    pos = colptr[j]
    for i = j:min((k+1)*p,j+p)
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, n, (k+1)*p)
  U = zeros(FC, m, (k+1)*p)
  L = SparseMatrixCSC((k+1)*p, (k+1)*p, colptr, rowval, nzval)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, n, p)
  qᵤ = zeros(FC, m, p)
  Ψ₁ = zeros(FC, p, p)
  Ψᵢ₊₁ = TΩᵢ = TΩᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      qᵤ .= B
      reduced_qr!(qᵤ, Ψ₁, algo)
      uᵢ .= qᵤ

      mul!(qᵥ, Aᴴ, uᵢ)
      reduced_qr!(qᵥ, TΩᵢ, algo)
      vᵢ .= qᵥ

      # Store the block Ω₁ in Lₖ₊₁.ₖ₊₁
      for ii = 1:p
        indi = pos1+ii-1
        for jj = 1:p
          indj = pos1+jj-1
          (ii ≤ jj) && (L[indj,indi] = conj(TΩᵢ[ii,jj]))
        end
      end
    end

    mul!(qᵤ, A, vᵢ)
    mul!(qᵤ, uᵢ, TΩᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢ

    reduced_qr!(qᵤ, Ψᵢ₊₁, algo)
    uᵢ₊₁ .= qᵤ

    # Store the block Ψᵢ₊₁ in Lₖ₊₁.ₖ₊₁
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (L[indi,indj] = Ψᵢ₊₁[ii,jj])
      end
    end

    mul!(qᵥ, Aᴴ, uᵢ₊₁)
    mul!(qᵥ, vᵢ, Ψᵢ₊₁', α, β)  # qᵥ = qᵥ - vᵢ * Ψᵢ₊₁ᴴ

    reduced_qr!(qᵥ, TΩᵢ₊₁, algo)
    vᵢ₊₁ .= qᵥ

    # Store the block Ωᵢ₊₁ in Lₖ₊₁.ₖ₊₁
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos3+jj-1
        (ii ≤ jj) && (L[indj,indi] = conj(TΩᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, U, Ψ₁, L
end

"""
    V, Ψ, T, U, Φᴴ, Tᴴ = saunders_simon_yip(A, B, C, k; algo="householder")

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a matrix of size `m × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Saunders-Simon-Yip process.

#### Keyword argument

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).

#### Output arguments

* `V`: a dense `m × p(k+1)` matrix;
* `Ψ`: a dense `p × p` upper triangular matrix such that `V₁Ψ = B`;
* `T`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Φᴴ`: a dense `p × p` upper triangular matrix such that `U₁Φᴴ = C`;
* `Tᴴ`: a sparse `p(k+1) × pk` block tridiagonal matrix with a bandwidth `p`.
"""
function saunders_simon_yip(A, B::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int; algo::String="householder") where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(B)
  Aᴴ = A'

  nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
  colptr = zeros(Int, p*k+1)
  rowval = zeros(Int, nnzT)
  nzval_T = zeros(FC, nnzT)
  nzval_Tᴴ = zeros(FC, nnzT)

  colptr[1] = 1
  for j = 1:k*p
    pos = colptr[j]
    for i = max(1, j-p):j+p
      rowval[pos] = i
      pos += 1
    end
    colptr[j+1] = pos
  end

  V = zeros(FC, m, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  T = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_T)
  Tᴴ = SparseMatrixCSC((k+1)*p, k*p, colptr, rowval, nzval_Tᴴ)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, m, p)
  qᵤ = zeros(FC, n, p)
  Ψ₁ = zeros(FC, p, p)
  Φ₁ᴴ = zeros(FC, p, p)
  Ωᵢ = Ψᵢ = Ψᵢ₊₁ = TΦᵢ = TΦᵢ₊₁ = zeros(FC, p, p)

  for i = 1:k
    pos1 = (i-1)*p + 1
    pos2 = i*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vᵢ = view(V,:,pos1:pos2)
    vᵢ₊₁ = view(V,:,pos3:pos4)
    uᵢ = view(U,:,pos1:pos2)
    uᵢ₊₁ = view(U,:,pos3:pos4)

    if i == 1
      qᵥ .= B
      reduced_qr!(qᵥ, Ψ₁, algo)
      vᵢ .= qᵥ
      qᵤ .= C
      reduced_qr!(qᵤ, Φ₁ᴴ, algo)
      uᵢ .= qᵤ
    end

    mul!(qᵥ, A, uᵢ)
    mul!(qᵤ, Aᴴ, vᵢ)

    if i ≥ 2
      pos5 = pos1 - p
      pos6 = pos2 - p
      vᵢ₋₁ = view(V,:,pos5:pos6)
      uᵢ₋₁ = view(U,:,pos5:pos6)

      mul!(qᵥ, vᵢ₋₁, TΦᵢ', α, β)  # qᵥ = qᵥ - vᵢ₋₁ * Φᵢ
      for ii = 1:p
        indi = pos1+ii-1
        for jj = 1:p
          indj = pos5+jj-1
          Ψᵢ[ii,jj] = T[indi,indj]
        end
      end
      mul!(qᵤ, uᵢ₋₁, Ψᵢ', α, β)  # qᵤ = qᵤ - uᵢ₋₁ * Ψᵢᴴ
    end

    mul!(Ωᵢ, vᵢ', qᵥ)
    mul!(qᵥ, vᵢ, Ωᵢ, α, β)   # qᵥ = qᵥ - vᵢ * Ωᵢ
    mul!(qᵤ, uᵢ, Ωᵢ', α, β)  # qᵤ = qᵤ - uᵢ * Ωᵢᴴ

    # Store the block Ωᵢ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos1+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        T[indi,indj] = Ωᵢ[ii,jj]
        Tᴴ[indi,indj] = conj(Ωᵢ[jj,ii])
      end
    end

    reduced_qr!(qᵥ, Ψᵢ₊₁, algo)
    vᵢ₊₁ .= qᵥ

    # Store the block Ψᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (T[indi,indj] = Ψᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (Tᴴ[indj,indi] = conj(Ψᵢ₊₁[ii,jj]))
      end
    end

    reduced_qr!(qᵤ, TΦᵢ₊₁, algo)
    uᵢ₊₁ .= qᵤ

    # Store the block Φᵢ₊₁ in Tₖ₊₁.ₖ
    for ii = 1:p
      indi = pos3+ii-1
      for jj = 1:p
        indj = pos1+jj-1
        (ii ≤ jj) && (Tᴴ[indi,indj] = TΦᵢ₊₁[ii,jj])
        (ii ≤ jj) && (i < k) && (T[indj,indi] = conj(TΦᵢ₊₁[ii,jj]))
      end
    end
  end
  return V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ
end

"""
    V, Γ, H, U, Λ, F = montoison_orban(A, B, D, C, k; algo="householder", reorthogonalization=false)

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `B`: a linear operator that models a matrix of dimension `n × m`;
* `D`: a matrix of size `m × p`;
* `C`: a matrix of size `n × p`;
* `k`: the number of iterations of the block Montoison-Orban process.

#### Keyword arguments

* `algo`: the algorithm to perform reduced QR factorizations (`"gs"`, `"mgs"`, `"givens"` or `"householder"`).
* `reorthogonalization`: reorthogonalize the new matrices of the block Krylov basis against all previous matrices.

#### Output arguments

* `V`: a dense `m × p(k+1)` matrix;
* `Γ`: a dense `p × p` upper triangular matrix such that `V₁Γ = D`;
* `H`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`;
* `U`: a dense `n × p(k+1)` matrix;
* `Λ`: a dense `p × p` upper triangular matrix such that `U₁Λ = C`;
* `F`: a dense `p(k+1) × pk` block upper Hessenberg matrix with a lower bandwidth `p`.
"""
function montoison_orban(A, B, D::AbstractMatrix{FC}, C::AbstractMatrix{FC}, k::Int; algo::String="householder", reorthogonalization::Bool=false) where FC <: FloatOrComplex
  m, n = size(A)
  t, p = size(D)

  V = zeros(FC, m, (k+1)*p)
  U = zeros(FC, n, (k+1)*p)
  H = zeros(FC, (k+1)*p, k*p)
  F = zeros(FC, (k+1)*p, k*p)

  α = -one(FC)
  β = one(FC)
  qᵥ = zeros(FC, m, p)
  qᵤ = zeros(FC, n, p)
  Γ = zeros(FC, p, p)
  Λ = zeros(FC, p, p)
  Ψᵢⱼ = Φᵢⱼ = Ψtmp = Φtmp = zeros(FC, p, p)

  for j = 1:k
    pos1 = (j-1)*p + 1
    pos2 = j*p
    pos3 = pos1 + p
    pos4 = pos2 + p
    vⱼ = view(V,:,pos1:pos2)
    vⱼ₊₁ = view(V,:,pos3:pos4)
    uⱼ = view(U,:,pos1:pos2)
    uⱼ₊₁ = view(U,:,pos3:pos4)

    if j == 1
      qᵥ .= D
      reduced_qr!(qᵥ, Γ, algo)
      vⱼ .= qᵥ

      qᵤ .= C
      reduced_qr!(qᵤ, Λ, algo)
      uⱼ .= qᵤ
    end

    mul!(qᵥ, A, uⱼ)
    mul!(qᵤ, B, vⱼ)

    for i = 1:j
      pos5 = (i-1)*p + 1
      pos6 = i*p
      vᵢ = view(V,:,pos5:pos6)
      uᵢ = view(U,:,pos5:pos6)

      mul!(Ψᵢⱼ, vᵢ', qᵥ)       # Ψᵢⱼ = vᵢᴴ * qᵥ
      mul!(qᵥ, vᵢ, Ψᵢⱼ, α, β)  # qᵥ = qᵥ - vᵢ * Ψᵢⱼ
      H[pos5:pos6,pos1:pos2] .= Ψᵢⱼ

      mul!(Φᵢⱼ, uᵢ', qᵤ)       # Φᵢⱼ = uᵢᴴ * qᵤ
      mul!(qᵤ, uᵢ, Φᵢⱼ, α, β)  # qᵤ = qᵤ - uᵢ * Φᵢⱼ
      F[pos5:pos6,pos1:pos2] .= Φᵢⱼ
    end

    if reorthogonalization
      for i = 1:j
        pos5 = (i-1)*p + 1
        pos6 = i*p
        vᵢ = view(V,:,pos5:pos6)
        uᵢ = view(U,:,pos5:pos6)

        mul!(Ψtmp, vᵢ', qᵥ)       # Ψtmp = vᵢᴴ * qᵥ
        mul!(qᵥ, vᵢ, Ψtmp, α, β)  # qᵥ = qᵥ - vᵢ * Ψtmp
        Hᵢⱼ = view(H,pos5:pos6,pos1:pos2)
        Hᵢⱼ .= Hᵢⱼ .+ Ψtmp

        mul!(Φtmp, uᵢ', qᵤ)       # Φtmp = uᵢᴴ * qᵤ
        mul!(qᵤ, uᵢ, Φtmp, α, β)  # qᵤ = qᵤ - uᵢ * Φtmp
        Fᵢⱼ = view(F,pos5:pos6,pos1:pos2)
        Fᵢⱼ .= Fᵢⱼ .+ Φtmp
      end
    end

    Ψ = view(H, pos3:pos4,pos1:pos2)
    reduced_qr!(qᵥ, Ψ, algo)
    vⱼ₊₁ .= qᵥ

    Φ = view(F, pos3:pos4,pos1:pos2)
    reduced_qr!(qᵤ, Φ, algo)
    uⱼ₊₁ .= qᵤ
  end
  return V, Γ, H, U, Λ, F
end
