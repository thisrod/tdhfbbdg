# shift and invert

using LinearAlgebra, Plots, Optim, Arpack, LinearMaps

using Revise
using Superfluids

default(:legend, :none)
default(:aspect_ratio, 1)

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FDDiscretisation{2}(30, 15/29)
g_tol = 1e-7

L, H, J = Superfluids.operators(s,d,:L,:H,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(2μ)

function rsdl2(q, Ω)
    Lq = L(q;Ω)
    μ = dot(Lq,q)
    sum(abs2, Lq-μ*q)
end

Ω = 0.3
# Optimizing E gives better answer than optimizing residual
result = optimize(0.2R_TF, 0.5R_TF, abs_tol=g_tol) do r
    q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)
    real(dot(H(q;Ω),q))
end
r = result.minimizer

q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)

# ws, us, vs = bdg_modes(s, d, q, Ω, 10, nev=100)

# B = Superfluids.BdGmatrix(s,d,Ω,q)
# ew, ev = eigs(B, nev=20, which=:SM)

B = Superfluids.bdg_operator(s,d,q,Ω)
Bop = LinearMap{Complex{Float64}}(2d.n^2) do uv
    u = uv[1:d.n^2]
    u = reshape(u, d.n, d.n)
    v = uv[d.n^2+1:end]
    v = reshape(v, d.n, d.n)
    uu, vv = B(u,v)
    [uu[:]; vv[:]]
end
# use pairplots.jl

LinearAlgebra.factorize(A::LinearMaps.LinearMap) = A

ew, ev = eigs(Bop; nev=20, which=:SM)[1:2]