```@meta
# Eigenvalue problems
```

## Krylov-Schur

```@docs
krylov_schur
krylov_schur!
KrylovSchurWorkspace
```

## Shift-invert

Eigenvalues in the interior of the spectrum, or close to the imaginary axis, converge slowly with
`which = :LR` or `which = :SM`.
The shift-invert spectral transformation maps the eigenvalues `λ` of `A` closest to a shift `σ`
to the eigenvalues `μ = 1 / (λ - σ)` of largest magnitude of `(A - σI)⁻¹`, which converge in a few cycles.
`krylov_schur` only needs `size`, `eltype` and `mul!` for the operator, so a factorization of `A - σI`
can be wrapped in a small type whose `mul!` method calls `ldiv!`.
The shift `σ` can be complex, in which case the starting vector must be complex too.

```@example shift_invert
using Krylov, LinearAlgebra, SparseArrays

# Two-dimensional convection-diffusion operator
N = 30
h = 1 / (N + 1)
D = spdiagm(-1 => ones(N-1), 0 => -2 * ones(N), 1 => ones(N-1)) / h^2
C = spdiagm(-1 => -ones(N-1), 1 => ones(N-1)) / (2h)
Iₙ = sparse(1.0I, N, N)
A = kron(Iₙ, D) + kron(D, Iₙ) + 20 * kron(Iₙ, C) + 5 * kron(C, Iₙ)

struct ShiftInvert{FC,F}
  F :: F
end
ShiftInvert(F, FC) = ShiftInvert{FC,typeof(F)}(F)
Base.size(op::ShiftInvert) = size(op.F)
Base.eltype(::ShiftInvert{FC}) where FC = FC
LinearAlgebra.mul!(y, op::ShiftInvert, x) = ldiv!(y, op.F, x)

σ = -100.0 + 50.0im
op = ShiftInvert(lu(A - σ * I), ComplexF64)
μ, X, stats = krylov_schur(op, ones(ComplexF64, N^2); nev = 4, memory = 20)
λ = σ .+ 1 ./ μ
```

```@example shift_invert
stats
```

```@example shift_invert
norm(A * X[:,1] - λ[1] * X[:,1])
```

For the generalized eigenvalue problem `Ax = λMx`, the operator is `(A - σM)⁻¹M`, whose `mul!`
applies `M` and then the factorization of `A - σM`.
When a factorization is too expensive, `mul!` can call an inner Krylov method such as [`gmres!`](@ref)
with a preconditioner; the inner tolerance must then be tighter than the tolerance of the eigenvalues.
