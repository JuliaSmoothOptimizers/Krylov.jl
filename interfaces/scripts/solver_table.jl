# solver_table.jl — single source of truth for the LibKrylov solver list.
# Included by both generate_stores.jl and generate_header.jl.
#
# Each entry: (c_name, workspace_type_name, enum_constant)
# The enum value equals the 0-based index in this list (must stay stable).

const SOLVERS = [
  ("cg",         "CgWorkspace",       "KRYLOV_CG"),
  ("cr",         "CrWorkspace",       "KRYLOV_CR"),
  ("symmlq",     "SymmlqWorkspace",   "KRYLOV_SYMMLQ"),
  ("minres",     "MinresWorkspace",   "KRYLOV_MINRES"),
  ("minres_qlp", "MinresQlpWorkspace","KRYLOV_MINRES_QLP"),
  ("diom",       "DiomWorkspace",     "KRYLOV_DIOM"),
  ("dqgmres",    "DqgmresWorkspace",  "KRYLOV_DQGMRES"),
  ("fom",        "FomWorkspace",      "KRYLOV_FOM"),
  ("gmres",      "GmresWorkspace",    "KRYLOV_GMRES"),
  ("fgmres",     "FgmresWorkspace",   "KRYLOV_FGMRES"),
  ("bicgstab",   "BicgstabWorkspace", "KRYLOV_BICGSTAB"),
  ("cgs",        "CgsWorkspace",      "KRYLOV_CGS"),
  ("bilq",       "BilqWorkspace",     "KRYLOV_BILQ"),
  ("qmr",        "QmrWorkspace",      "KRYLOV_QMR"),
  ("usymlq",     "UsymlqWorkspace",   "KRYLOV_USYMLQ"),
  ("usymqr",     "UsymqrWorkspace",   "KRYLOV_USYMQR"),
  ("tricg",      "TricgWorkspace",    "KRYLOV_TRICG"),
  ("trimr",      "TrimrWorkspace",    "KRYLOV_TRIMR"),
  ("trilqr",     "TrilqrWorkspace",   "KRYLOV_TRILQR"),
  ("bilqr",      "BilqrWorkspace",    "KRYLOV_BILQR"),
  ("lslq",       "LslqWorkspace",     "KRYLOV_LSLQ"),
  ("lsqr",       "LsqrWorkspace",     "KRYLOV_LSQR"),
  ("lsmr",       "LsmrWorkspace",     "KRYLOV_LSMR"),
  ("usymlqr",    "UsymlqrWorkspace",  "KRYLOV_USYMLQR"),
  ("cgls",       "CglsWorkspace",     "KRYLOV_CGLS"),
  ("crls",       "CrlsWorkspace",     "KRYLOV_CRLS"),
  ("cgne",       "CgneWorkspace",     "KRYLOV_CGNE"),
  ("crmr",       "CrmrWorkspace",     "KRYLOV_CRMR"),
  ("craig",      "CraigWorkspace",    "KRYLOV_CRAIG"),
  ("craigmr",    "CraigmrWorkspace",  "KRYLOV_CRAIGMR"),
  ("lnlq",       "LnlqWorkspace",     "KRYLOV_LNLQ"),
  ("gpmr",       "GpmrWorkspace",     "KRYLOV_GPMR"),
  ("car",        "CarWorkspace",      "KRYLOV_CAR"),
  ("minares",    "MinaresWorkspace",  "KRYLOV_MINARES"),
]

# Block Krylov solvers — a separate, matrix-based API (m×p block right-hand side).
# Each entry: (c_name, workspace_type_name, enum_constant).
# The enum value equals the 0-based index in this list.
const BLOCK_SOLVERS = [
  ("block_gmres",  "BlockGmresWorkspace",  "KRYLOV_BLOCK_GMRES"),
  ("block_minres", "BlockMinresWorkspace", "KRYLOV_BLOCK_MINRES"),
]
