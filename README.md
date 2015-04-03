# Krylov.jl: A Julia basket of hand-picked Krylov methods

[![Build Status](https://travis-ci.org/optimizers/Krylov.jl.svg)](https://travis-ci.org/optimizers/Krylov.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ygy8aqqjmqhfwrxc?svg=true)](https://ci.appveyor.com/project/dpo/krylov-jl)
[![Coverage Status](https://coveralls.io/repos/optimizers/Krylov.jl/badge.svg)](https://coveralls.io/r/optimizers/Krylov.jl)

## How to Install

At the Julia prompt, type

````JULIA
julia> Pkg.clone("https://github.com/dpo/LinearOperators.jl.git")
julia> Pkg.clone("https://github.com/optimizers/Krylov.jl.git")
julia> Pkg.test("Krylov")
````

## Long-Term Goals

* provide implementations of certain of the most useful Krylov method for
  linear systems with special emphasis on methods for linear least-squares
  problems and saddle-point linear system (including symmetric quasi-definite
  systems)
* provide state-of-the-art implementations alongside simple implementations of
  equivalent methods in exact artithmetic (e.g., LSQR vs. CGLS, MINRES vs. CR,
  LSMR vs. CRLS, etc.)
* provide simple, consistent calling signatures and avoid over-typing
* ensure those implementations are fast and stable.

This content is released under the [MIT](http://opensource.org/licenses/MIT) License.
<a rel="license" href="http://opensource.org/licenses/MIT">
<img alt="MIT license" height="40" src="http://upload.wikimedia.org/wikipedia/commons/c/c3/License_icon-mit.svg" /></a>
