# TODO List

## Methods

- [X] CG
- [X] CRLS: CR on A'Ax = A'b (equivalent to LSMR in exact arithmetic)
- [X] LSQR (equivalent to CG on A'Ax = A'b)
- [X] LSMR (equivalent to MINRES on A'Ax = A'b)
- [X] MINRES
- [X] CRAIG (equivalent to CG on AA'y=b, x = A'y)
- [X] CRAIGMR (must be equivalent to CR on AA'y = b, x = A'y)

## Special features

- [ ] Estimate norms and conditions numbers
- [X] Allow for regularization
- [X] Allow for shifts

## Preconditioning

- [ ] Save on storage when not preconditioning (How? Assign identifiers to ops)

## Stopping criterion

- [ ] Shifted systems stopped according to the paper
- [ ] cg interrupted in the spirit of "truncated Newton" with line search