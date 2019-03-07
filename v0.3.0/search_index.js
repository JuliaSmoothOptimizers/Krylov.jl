var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Home-1",
    "page": "Home",
    "title": "Krylov.jl documentation",
    "category": "section",
    "text": "This package implements iterative methods for the solution of linear systems of equations  Ax = blinear least-squares problems  min b - Axand linear least-norm problems  min x quad textsubject to  Ax = bIt is appropriate, in particular, in situations where such a problem must be solved but a factorization is not possible, either because:the operator is not available explicitly,\nthe operator is dense, or\nfactors would consume an excessive amount of memory and/or disk space.Iterative methods are particularly appropriate in either of the following situations:the problem is sufficiently large that a factorization is not feasible or would be slower,\nan effective preconditioner is known in cases where the problem has unfavorable spectral structure,\nthe operator can be represented efficiently as a sparse matrix,\nthe operator is fast, i.e., can be applied with far better complexity than if it were materialized as a matrix. Often, fast operators would materialize as dense matrices."
},

{
    "location": "#Objective:-solve-Ax-\\approx-b-1",
    "page": "Home",
    "title": "Objective: solve Ax approx b",
    "category": "section",
    "text": "Given a linear operator A and a right-hand side b, solve Ax = b, which means:when A has full column rank and b lies in the range space of A, find the unique x such that Ax = b; this situation occurs when\nA is square and nonsingular, or\nA is tall and has full column rank and b lies in the range of A,\nwhen A is column-rank deficient but b is in the range of A, find x with minimum norm such that Ax = b; this situation occurs when b is in the range of A and\nA is square but singular, or\nA is short and wide,\nwhen b is not in the range of A, regardless of the shape and rank of A, find x that minimizes the residual b - Ax. If there are infinitely many such x (because A is rank deficient), identify the one with minimum norm."
},

{
    "location": "#How-to-Install-1",
    "page": "Home",
    "title": "How to Install",
    "category": "section",
    "text": "Krylov can be installed and tested through the Julia package manager:julia> Pkg.add(\"Krylov\")\njulia> Pkg.test(\"Krylov\")"
},

{
    "location": "#Long-Term-Goals-1",
    "page": "Home",
    "title": "Long-Term Goals",
    "category": "section",
    "text": "provide implementations of certain of the most useful Krylov method for linear systems with special emphasis on methods for linear least-squares problems and saddle-point linear system (including symmetric quasi-definite systems)\nprovide state-of-the-art implementations alongside simple implementations of equivalent methods in exact artithmetic (e.g., LSQR vs. CGLS, MINRES vs. CR, LSMR vs. CRLS, etc.)\nprovide simple, consistent calling signatures and avoid over-typing\nensure those implementations are fast and stable."
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "Pages = [\"api.md\"]"
},

{
    "location": "api/#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": "KrylovStats\nSimpleStats\nLanczosStats\nSymmlqStats"
},

{
    "location": "api/#Utilities-1",
    "page": "API",
    "title": "Utilities",
    "category": "section",
    "text": "roots_quadratic\nsym_givens\nto_boundary\nvec2str"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

]}
