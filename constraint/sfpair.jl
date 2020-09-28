# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FDDiscretisation{2}(66, 0.3)
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
rr = range(d.h, R_TF, length=15)
qs = [steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)
    for r = rr]
lEs = [real(dot(H(q),q)) for q in qs]
rEs = [real(dot(H(q;Ω),q)) for q in qs]
rdls = rsdl2.(qs, Ω)

plot(
    scatter(rr/R_TF, rEs.-E₀, xshowaxis=false, ylabel="E (rot)",
        title="vortex pair, rotating frame W=0.3"),
    scatter(rr/R_TF, rdls, xlabel="r/R_TF", ylabel="residual"),
    layout=@layout [a;b]
)

# Optimizing E gives better answer than optimizing residual
result = optimize(0.2R_TF, 0.5R_TF, abs_tol=g_tol) do r
    q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)
    real(dot(H(q;Ω),q))
end
r = result.minimizer

q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)

ws, us, vs = bdg_modes(s, d, q, Ω, 10, nev=100)

# B = Superfluids.BdGmatrix(s,d,Ω,q)
# ew, ev = eigs(B, nev=20, which=:SM)


# use pairplots.jl
