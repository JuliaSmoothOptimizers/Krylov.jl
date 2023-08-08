module Krylov

using LinearAlgebra, SparseArrays, Printf

@static if !isdefined(Base, :get_extension)
    using Requires
end

include("krylov_utils.jl")
include("krylov_stats.jl")
include("krylov_solvers.jl")
include("krylov_processes.jl")

include("cg.jl")
include("cr.jl")

include("symmlq.jl")
include("cg_lanczos.jl")
include("cg_lanczos_shift.jl")
include("minres.jl")
include("minres_qlp.jl")

include("diom.jl")
include("fom.jl")
include("dqgmres.jl")
include("gmres.jl")
include("fgmres.jl")

include("gpmr.jl")

include("usymlq.jl")
include("usymqr.jl")
include("tricg.jl")
include("trimr.jl")
include("trilqr.jl")

include("cgs.jl")
include("bicgstab.jl")

include("bilq.jl")
include("qmr.jl")
include("bilqr.jl")

include("cgls.jl")
include("crls.jl")

include("cgne.jl")
include("crmr.jl")

include("lslq.jl")
include("lsqr.jl")
include("lsmr.jl")

include("lnlq.jl")
include("craig.jl")
include("craigmr.jl")

include("krylov_solve.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("../ext/KrylovStaticArraysExt.jl")
    end
end

end
