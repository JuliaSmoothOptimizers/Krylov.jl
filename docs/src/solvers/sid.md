```@meta
# Hermitian indefinite linear systems
```

## SYMMLQ

```@docs
symmlq
symmlq!
SymmlqWorkspace
```

## MINRES

```@docs
minres
minres!
MinresWorkspace
```

## MINRES-QLP

```@docs
minres_qlp
minres_qlp!
MinresQlpWorkspace
```

## MINARES

```@docs
minares
minares!
MinaresWorkspace
```

## Complex symmetric linear systems

CS-MinAres is for complex symmetric matrices (`transpose(A) == A`), a
different structure from the Hermitian (indefinite) matrices above
(`A' == A`); the two coincide only for real symmetric matrices.

### CS-MINARES

```@docs
csminares
csminares!
CsMinaresWorkspace
```
