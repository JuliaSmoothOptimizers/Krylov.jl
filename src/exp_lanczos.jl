# An implementation of exp(τA)v for Hermitian operators A.
#
# exp(τA)v = β₀Vₘ exp(τTₘ)e₁
#
# The implementation follows
# Saad, Y. (1992). Analysis of some krylov subspace
# approximations. SIAM Journal on Numerical Analysis.
#
# The error estimates/stopping criteria are borrowed from
# cg_lanczos.jl
#
# Stefanos Carlström, <stefanos.carlstrom@gmail.com>
# Lund, Sweden, May 2016.

export exp_lanczos, exp_lanczos!, exp_lanczos_work

# Check arguments and allocate work arrays
function exp_lanczos_work{T<:Number}(A :: LinearOperator,
                                     v :: Array{T,1},
                                     m :: Integer,
                                     verbose :: Bool=false)
    n = size(v, 1);
    (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size, $(size(A))×$(size(v))");
    verbose && @printf("exp Lanczos: %d×%d system\n", n, n);

    V = zeros(T, n, m+1)
    α = zeros(real(T), m)
    β = zeros(real(T), m)

    V,α,β
end

function expT(α,β,v,τ,β₀)
    ee = eigfact(SymTridiagonal(α,β))
    β₀*v*ee[:vectors]*Diagonal(exp(τ*ee[:values]))*ee[:vectors][1,:]'
end

function exp_lanczos!{T<:Number,R<:Real}(A :: LinearOperator,
                                               v :: AbstractVector{T},
                                               τ :: T, m :: Integer,
                                               vp :: AbstractVector{T},
                                               V :: Matrix{T},
                                               α :: Vector{R},
                                               β :: Vector{R};
                                               atol :: Float64=1.0e-8,
                                               rtol :: Float64=1.0e-4,
                                               verbose :: Bool=false)
    β₀ = norm(v)
    V[:,1] = v/β₀
    
    ε = atol + rtol * β₀
    verbose && @printf("Initial norm: β₀ %g, stopping threshold: %g\n", β₀, ε)

    fin = false
    j = 1
    jj = 1 # Which Krylov subspace to use, in the end
    
    σ = β₀
    ω = 0
    γ = 1

    for j = 1:m
        V[:,j+1] = A*V[:,j]
        j > 1 && (V[:,j+1] -= β[j-1]*V[:,j-1])
        α[j] = real(dot(V[:,j],V[:,j+1]))
        
        γ = 1 / (α[j] - ω / γ)
        γ <= 0.0 && break

        V[:,j+1] -= α[j]*V[:,j]
        β[j] = norm(V[:,j+1])
        V[:,j+1] /= β[j]
        
        ω = β[j] * γ
        σ = -ω * σ
        ω = ω * ω
        
        verbose && @printf("iter %d, α[%d] %g, β[%d] %g, γ %g, ω %g, σ %g\n",
                          j, j, α[j], j, β[j],
                          γ, ω, σ)

        if abs(σ) < ε
            break
        else
            j != m && (jj += 1)
        end
    end
    verbose && println("Krylov subspace size: ", jj)
    vp[:] = expT(α[1:jj], β[1:jj-1], V[:,1:jj], τ, β₀)
end

# One-shot version
function exp_lanczos{T<:Number}(A :: LinearOperator, v :: Array{T,1},
                                τ :: T, m :: Integer;
                                atol :: Float64=1.0e-8,
                                rtol :: Float64=1.0e-6,
                                verbose :: Bool=false)
    l_work = exp_lanczos_work(A, v, m, verbose)
    vp = similar(v)
    exp_lanczos!(A, v, τ, m, vp, l_work...; βtol = βtol, verbose = verbose)
    vp
end
