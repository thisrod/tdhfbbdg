# ground state and GPE dynamics for a harmonic trap with a moat

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, Statistics, JLD2
using Plots, ComplexPhasePortrait

results = "E.jld2"

# @load results C W R y w steps source
@load results C W y steps source
Ω = W

N = length(y)
h = (y[end] - y[1])/(N-1)
x = y';  z = x .+ 1im*y
r = abs.(z)
r² = abs2.(z)

# mt = @. exp(-(r-R)^2/2/w^2)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂ = (1/h).*op([0, -1/2, 0, 1/2, 0])
∂² = (1/h^2).*op(Float64[0, 0, 1, -2, 1, 0, 0])

# kludge BC
∂[1,:] .= ∂[2,:]
∂[end,:] .= ∂[end-1,:]
∂²[1,:] .= ∂²[2,:]
∂²[end,:] .= ∂²[end-1,:]

# fudge phase

function fudge(u)
    hat = mean(@. u*(abs(z)>2))
    hat /= abs(hat)
    u ./ (norm(u)*hat)
end

# plot(zplot(ψ), scatter(abs.(z[:]), abs.(fft(ψ)[:]), yscale=:log10, ms=2, mc=:black, msw=0, leg=:none), layout = @layout [a b])

# hh = 2π*(0:0.01:1)
# plot!(R*sin.(hh), R*cos.(hh), lc=:white, leg=:none)

function load_moat(j)
    jldopen(results, "r") do file
        file["psi$(j)"], file["t$(j)"]
    end
end

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))

function poles(u)
    st = [-h 0 h]
    rs = st .+ 1im*st'
    v = u ./ abs.(u)
    A = [rs.*v[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    B = [conj.(rs).*v[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    # return the difference mode of vortices and antivortices, 
    P = @. (B-A)*(abs(A)<abs(B))
    Q = @. (B-A)*(abs(A)>abs(B))
    P,Q
end

function locmax(u, ix::CartesianIndex)
    j, k = Tuple(ix)
    1 < j < size(u,1) && 1 < k < size(u,2) &&
        real(u[j,k]) > 0.1 &&
        real(u[j,k]) ≈ maximum(real, u[j-1:j+1, k-1:k+1])
end

locmax(u) = [k for k in keys(u) if locmax(u, k)]

function zmax(u)
    # coordinates at cluster centres
    # figure out why this swaps real and imag parts
    S = cluster_adjacent(adjacent_index, locmax(u))
    [Complex(h.*(Tuple(sum(s)) ./ length(s) .- (N-1)/2)...) for s in S]
end

function show_vortices(u)
    P, Q = poles(u)
    function markup!(X, col)
        zin = z[2:end-1, 2:end-1]
        f!(R, sym) = scatter!(X, imag.(zmax(R)), real.(zmax(R)), m=sym, ms=1, mc=col, msw=0, leg=:none)
        f!(Q, :circle)
        f!(P, :xcross)
    end
    A, B = (heatmap(x[2:end-1], y[2:end-1], real.(reverse(v,dims=1)), aspect_ratio=1) for v in poles(u))
    C = zplot(u)
    markup!(C, :white)
    D = argplot(u)
    markup!(D, :black)
    plot(C, D, B, A, layout = @layout [a b; c d])
end

function cluster_adjacent(f, ixs)
    # function f determines adjacency
    clusters = Set()
    for i in ixs
        out = Set()
        ins = Set()
        for C in clusters
            if any(j -> f(i,j), C)
                push!(ins, C)
            else
                push!(out, C)
            end
        end
        push!(out, union(Set([i]), ins...))
        clusters = out
    end
    clusters
end

adjacent_index(j, k) =
    -1 ≤ j[1] - k[1] ≤ 1 && -1 ≤ j[2] - k[2] ≤ 1

# qslice = (ψ[N÷2,:] + ψ[N÷2+1,:])/2
# P1 = scatter(y, real.(qslice), ms=1.5, mc=:black, msw=0, leg=:none)
# xlabel!("x");  ylabel!("psi");  title!("Slice along x-axis")
# vslice = (V[N÷2,:] + V[N÷2+1,:])/2
# A = 10.0ab(3.3, 0.1);  aslice = (A[N÷2,:] + A[N÷2+1,:])/2;
# P2 = scatter(y, aslice, ms=1.5, mc=:red, msw=0, leg=:none)
# scatter!(y, vslice, ms=1.5, mc=:black, msw=0, leg=:none)
# plot!(xlims() |> collect, [m, m], lc=:blue, lw=2)
# ylabel!("V, absn red, mu blue")

function ab(rvac, bord)
    out = @. exp(-(r-rvac)^2/2/bord^2)
    out[r .> rvac] .= 1
    out
end

function show_slice(j)
    q = load_moat(j) |> first
    qslice = (q[N÷2,:] + q[N÷2+1,:])/2
    U, S, _ = svd([real.(qslice) imag.(qslice)])
    s = sign(sum(U[:,1]))
    P1 = scatter(y, s*S[1]*U[:,1], ms=1, mc=:black, msw=0, leg=:none)
    if S[2] > 1e-3*S[1]
        scatter!(y, s*S[2]*U[:,2], ms=1, mc=:red, msw=0, leg=:none)
    end
    xlabel!("x");  ylabel!("psi");  title!("Slice along x-axis")
    vslice = (V[N÷2,:] + V[N÷2+1,:])/2
    A = 10.0ab(3.3, 0.1);  aslice = (A[N÷2,:] + A[N÷2+1,:])/2
    P2 = scatter(y, aslice, ms=1, mc=:red, msw=0, leg=:none)
    scatter!(y, vslice, ms=1, mc=:black, msw=0, leg=:none)
    L = -(∂²*q+q*∂²')/2+V.*q+C/2h*abs2.(q).*q
    m = sum(conj.(q).*L) |> real
    plot!(xlims() |> collect, [m, m], lc=:blue, lw=2)
    ylabel!("V, absn red, mu blue")
    plot(P1, P2, layout = @layout[a;b])
end

function berry_phases()
    φ = Float64[]
    tt = Float64[]
    q1, t1 = load_moat(0)
    Q1 = fudge(q1)
    for j = 1:steps
        q2, t2 = load_moat(j)
        Q2 = fudge(q2)
        push!(φ, sum(@. conj(Q2)*Q1 |> imag))
        push!(tt, t2)
        Q1 = Q2
    end
    φ, tt
end

function berry_step(j)
    q1, t1 = load_moat(j)
    Q1 = fudge(q1)
    q2, t2 = load_moat(j+1)
    Q2 = fudge(q2)
    @. conj(Q2)*Q1 |> imag
end

function show_step(j)
    P1 = load_moat(j) |> first |> fudge |> argplot
    P2 = berry_step(j) |> zplot
    P3 = load_moat(j+1) |> first |> fudge |> argplot
    plot(P1, P2, P3, layout = @layout[a b c])
end

function pcis(k)
    φ = zeros(N, N)
    q1, t1 = load_moat(0)
    Q1 = fudge(q1)
    t2 = NaN
    for j = 1:k
        q2, t2 = load_moat(j)
        Q2 = fudge(q2)
        @. φ += conj(Q2)*Q1 |> imag
        Q1 = Q2
    end
    φ
end

function diagnostic(s="")
    ff, tt = berry_phases()
    nn = [norm(load_moat(j) |> first) for j = 1:steps]
    P1 = scatter(tt, nn, ms=3, mc=:black, msw=0, leg=:none)
    title!(s)
    ylabel!("norm(psi)")
    P2 = scatter(tt, cumsum(ff), ms=3, mc=:black, msw=0, leg=:none)
    xlabel!("t")
    ylabel!("berry phase")
    plot(P1, P2, layout = @layout [a; b])
end

# [l for l in split(source, '\n') if occursin("f(ψ,_,t) =", l)] |> first

show_vortices(j::Int) = load_moat(j) |> first |> show_vortices

function show_field(j)
    ff, tt = berry_phases()
    nn = [norm(load_moat(j) |> first) for j = 1:steps]
    P1 = scatter(tt, nn, ms=1, mc=:black, msw=0, leg=:none)
    scatter!(tt[j:j], nn[j:j], ms=2, mc=:red, msw=0, leg=:none)
    ylabel!("norm(psi)")
    F = cumsum(ff)
    P2 = scatter(tt, F, ms=1, mc=:black, msw=0, leg=:none)
    scatter!(tt[j:j], F[j:j], ms=2, mc=:red, msw=0, leg=:none)
    xlabel!("t")
    ylabel!("berry phase")
    P3 = berry_step(j) |> zplot
    title!("step")
    P4 = pcis(j) |> zplot
    title!("cumulative")
    L = plot(P1, P2, layout = @layout [a; b])
    q = load_moat(j) |> first
    plot(L, zplot(q), argplot(q), show_slice(j), P3, P4, layout = @layout [a [b; c] d [e; f]])
end