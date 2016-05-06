plot_p = "TEST_KRYLOV_PLOT" in keys(ENV)

if plot_p
    using PyCall
    pygui(:qt)
    using PyPlot
    fignum = 0
end

function diff2mat(N)
    dx = 1/(N+1)
    SymTridiagonal(-2ones(N),ones(N-1))/(dx^2)
end

import Base: -
    -(T::SymTridiagonal) = SymTridiagonal(-diag(T),-diag(T,1))

Ham(N) = -diff2mat(N)/2

function initial_state{T<:Number}(H::AbstractArray,
                                  c::Vector{T})
    d,v = eigs(H; nev = length(c),
               which = :SM)

    c_norm = sqrt(sumabs2(c))
    c /= c_norm

    v0 = v*c
    E0 = dot(c.^2,d)

    N = size(H, 1)
    dx = 1/(N+1)

    # Return initial state, as well as spectrum
    E0,v0/(norm(v0)*sqrt(dx)),d,v/sqrt(dx),c
end

psi_norm(psi, dx) = norm(psi)*sqrt(dx)
tot_energy(H0, psi, dx) = real(dot(psi,H0*psi))*dx

Base.factorize{T<:Complex}(A::SymTridiagonal{T}) = factorize(Tridiagonal(diag(A,-1),diag(A), diag(A,1)))

function test_propagation(m, N, t, c = nothing;
                          verbose = false,
                          cn_dist_tol = 1e-7)
    dx = 1/(N+1)
    x = linspace(dx,1-dx, N)

    c == nothing && (c = [1])

    @time H0 = Ham(N)
    println("Finding initial state")
    @time E0,psi0,E_phi,phi,c = initial_state(H0, c)
    DE = Diagonal(E_phi)
    @printf("Initial energy: %e\n", E0)

    τ = -im*(t[2]-t[1])


    psi = begin
        psi = zeros(Complex128,N,length(t))
        psi[:,1] = psi0

        H0_op = LinearOperator(H0)
        println("Allocating work arrays")
        @time l_work = exp_lanczos_work(H0_op, psi[:,1], m, verbose)

        println("Propagating")
        @time for i = 2:length(t)
            do_print = (verbose || mod(i,div(length(t), 10)) == 0)
            do_print && println("----------------------")
            do_print && println("Step: $i")
            exp_lanczos!(H0_op, sub(psi, :, i-1), τ, m, sub(psi, :, i), l_work...;
                         rtol = 1e-5, verbose = do_print)
            do_print && println("----------------------")
        end
        
        psi
    end

    psi_cmp = begin
        # A three-point discretization of the 1D time-dependent
        # Schrödinger equation can be efficiently propagated using
        # Crank–Nicolson.
        I = Diagonal(ones(N))
        F = I + τ/2*H0
        B = factorize(I - τ/2*H0)
        psi_cmp = similar(psi)
        psi_cmp[:,1] = psi0
        println("Propagating using Crank–Nicolson")
        @time for i = 2:length(t)
            psi_cmp[:,i] = B \ (F*psi_cmp[:,i-1])
        end
        psi_cmp
    end

    if plot_p
        n = map(eachindex(t)) do j
            psi_norm(psi[:,j], dx)
        end
        n_cmp = map(eachindex(t)) do j
            psi_norm(psi_cmp[:,j], dx)
        end

        E = map(eachindex(t)) do j
            tot_energy(H0, psi[:,j], dx)
        end
        E_cmp = map(eachindex(t)) do j
            tot_energy(H0, psi_cmp[:,j], dx)
        end

        function time_series_plot{T<:Number}(v::Vector{T}, args...; kwargs...)
            av = abs(v)
            style = length(av) > 100 ? "-" : ".-"
            plot_f = any(av.>0) ? semilogy : plot
            plot_f(av, style, args...; kwargs...)

            margins(0,1)
        end

        function plot_state(i)
            l = plot(x, abs2(psi[:,i]), "-", label="Lanczos, $i")
            # plot(x, abs2(phi*exp(-im*t[i]*DE)*c), "o", color = l[1][:get_color]())
            plot(x, abs2(psi_cmp[:,i]), "x", color = l[1][:get_color](),
                 label="Crank–Nicolson, $i")
        end

        global fignum
        figure(fignum+=1)
        clf()
        subplot(221)
        for i = 1:4
            plot_state(div(i*length(t),4))
        end
        margins(0,0.1)
        legend(framealpha=0.75)
        title("State")
        subplot(243)
        pcolormesh(t, x, abs2(psi), rasterized=true, cmap = plt[:get_cmap]("viridis"))
        margins(0,0)
        xlabel("t")
        ylabel("x")
        title("Lanczos")
        subplot(244)
        pcolormesh(t, x, abs2(psi_cmp), rasterized=true, cmap = plt[:get_cmap]("viridis"))
        margins(0,0)
        xlabel("t")
        ylabel("x")
        title("Crank–Nicolson")
        subplot(223)
        time_series_plot(n-1, label="Lanczos")
        time_series_plot(n_cmp-1, label="Crank–Nicolson")
        time_series_plot(vec(sumabs2(psi-psi_cmp,1)*dx), label="Distance")
        xlabel("Step")
        legend(framealpha=0.75)
        title("Norm loss")
        subplot(224)
        time_series_plot(E-E0, label="Lanczos")
        time_series_plot(E_cmp-E0, label="Crank–Nicolson")
        legend(framealpha=0.75)
        margins(0,1)
        xlabel("Step")
        title("Total energy loss")
        tight_layout()
    end

    @test_approx_eq_eps tot_energy(H0, psi[:,end], dx) E0 1e-8
    @test_approx_eq_eps psi_norm(psi[:,end], dx) 1 1e-8
    @test_approx_eq_eps sumabs2(psi[:,end]-psi_cmp[:,end])*dx 0 cn_dist_tol
end

m = 15
N = max(m+1,101)
nt = 2001

# For propagation of a single eigenstate, the propagator is
# essentially exact
test_propagation(m, N, linspace(0,1,nt), [1]; verbose = false)
println()
println("+++++++++++++++++++++++++++++++")
println()
test_propagation(m, N, linspace(0,1,nt), [1,1]; verbose = false)
println()
println("+++++++++++++++++++++++++++++++")
println()
test_propagation(m, N, linspace(0,1,nt), [1,1,1]; verbose = false, cn_dist_tol = 2e-6)

plot_p && show()
