# CKrylov

App for building shared library of the [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) package.
All commands shown below are executed in the `Krylov.jl/app/` directory.

## Installation instructions

1. Download and install Julia (version 1.3.1 or newer).

2. Install `PackageCompiler`
    ```bash
    $ julia -e 'using Pkg; Pkg.add("PackageCompiler")'
    ```

3. Instantiate the current environment
    ```bash
    $ julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

    ```

4. Build the shared library
    ```bash
    $ julia --startup-file=no --project=build -e 'using Pkg; Pkg.instantiate()'
    $ julia --startup-file=no --project=build build/build.jl
    ```
    The header files will be located at `Krylov.jl/app/CKrylov/include`.
    The shared libraries will be located at `Krylov.jl/app/CKrylov/lib`.

    For more information on how to create library with [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl), take a look at the [documentation](https://julialang.github.io/PackageCompiler.jl/dev/libs)and the [presentation](https://www.youtube.com/watch?v=c0IAP7NC2MU) of JuliaCon 2021.
