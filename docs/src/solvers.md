All methods require that A is an `AbstractLinearOperator`.
But a variant allows you to give A as an `AbstractMatrix`. Thereafter A is automatically wrapped in a `LinearOperator`.

Detailed examples may be found [here](https://github.com/JuliaSmoothOptimizers/Krylov.jl/tree/master/examples).

## Krylov methods

```@docs
cg
cr
symmlq
cg_lanczos
cg_lanczos_shift_seq
minres
minres_qlp
diom
dqgmres
usymlq
usymqr
bilq
cgs
qmr
cgls
crls
cgne
crmr
lslq
lsqr
lsmr
craig
craigmr
```
