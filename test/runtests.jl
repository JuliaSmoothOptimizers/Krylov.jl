using Base.Test
using Krylov
using LinearOperators

include("gen_lsq.jl")
include("test_cg.jl")
include("test_cg_lanczos.jl")
include("test_cgls.jl")
include("test_crls.jl")
