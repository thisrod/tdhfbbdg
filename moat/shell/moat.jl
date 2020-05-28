# display shell for MO cluster runs

using LinearAlgebra, BandedMatrices, Optim, JLD2

using Plots, ComplexPhasePortrait

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))

C = 10_000.0
Ω = 0.0
R = 1.7
w = 0.1	# moat width
# ω = -2.86	# potential offset outside moat for lock step
# ω = -10.0	# potential offset outside moat for fast vortex
ω = 0.0

h = 0.05
N = 120
# h = 0.3
# N = 20


y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
r² = abs2.(z)

V = r²  + 20*R^2*exp.(-(r.-R).^2/2/w^2)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂ = (1/h).*op([0, -1/2, 0, 1/2, 0])
∂² = (1/h^2).*op(Float64[0, 0, 1, -2, 1, 0, 0])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴-Ω·ψ*Jψ
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
Ham(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/2h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(xy) = sum(conj.(togrid(xy)).*Ham(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

rc = 2	# condensate radius, 
P = Diagonal(sqrt.(rc^4 .+ V[:].^2))

@load "rmoat.jld2" result ψ

# hh = 2π*(0:0.01:1)
# plot!(R*sin.(hh), R*cos.(hh), lc=:white, leg=:none)

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
        real(u[j,k]) > 0.05 &&
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