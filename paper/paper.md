---
title: 'Krylov.jl: A Julia basket of hand-picked Krylov methods'
tags:
  - Julia
  - linear algebra
  - Krylov methods
  - sparse linear systems
authors:
  - name: Alexis Montoison^[corresponding author]
    orcid: 0000-0002-3403-5450
    affiliation: 1
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
   index: 1
date: 29 July 2022
bibliography: paper.bib

---

# Summary

`Krylov.jl` is a Julia [@bezanson-edelman-karpinski-shah-2017] package that provide implementations of certain of the most useful Krylov method for a variety of problems:

1 - Square or rectangular full-rank systems

$$ Ax = b $$

2 - Linear least-squares problems

$$ \min \|b - Ax\| $$

3 - Linear least-norm problems

$$ \min \|x\| \quad \text{subject to} \quad Ax = b $$

4 - Adjoint systems

$$ Ax = b \quad \text{and} \quad A^T y = c $$

5 - Saddle-point and symmetric quasi-definite systems

$$ \begin{bmatrix} M & \phantom{-}A \\ A^T & -N \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix} $$

6 - Generalized saddle-point and unsymmetric partitioned systems

$$ \begin{bmatrix} M & A \\ B & N \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ c \end{bmatrix} $$

# Statement of need

# Acknowledgements

Alexis Montoison is supported by a FRQNT grant and an excellence scholarship of the IVADO institute,
and Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
