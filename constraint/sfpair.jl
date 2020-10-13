# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Plots, Optim, Arpack, JLD2

using Revise
using Superfluids
using Superfluids: find_vortices, poles, cluster_adjacent, adjacent_index

default(:legend, :none)
Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
# d = FDDiscretisation{2}(66, 0.3, 7)
d = FDDiscretisation{2}(100, 0.2, 7)
g_tol = 1e-7

L, H, J = Superfluids.operators(s,d,:L,:H,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(2μ)
Ω = 0.3

function rsdl2(q, Ω)
    Lq = L(q;Ω)
    μ = dot(Lq,q)
    sum(abs2, Lq-μ*q)
end

# rr = range(d.h, R_TF, length=15)
# qs = [steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000) for r = rr]
# lEs = [real(dot(H(q),q)) for q in qs]
# rEs = [real(dot(H(q;Ω),q)) for q in qs]
# rdls = rsdl2.(qs, Ω)
# 
# plot(
#     scatter(rr/R_TF, rEs.-E₀, xshowaxis=false, ylabel="E (rot)",
#         title="vortex pair, rotating frame W=0.3"),
#     scatter(rr/R_TF, rdls, xlabel="r/R_TF", ylabel="residual"),
#     layout=@layout [a;b]
# )

function modes(d)
    H = Superfluids.operators(s,d,:H) |> only
    result = optimize(0.2R_TF, 0.5R_TF, abs_tol=g_tol) do r
        q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=5000)
        real(dot(H(q;Ω),q))
    end
    r = result.minimizer
    q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=5000)
    B = Superfluids.BdGmatrix(s,d,Ω,q)
    ew, ev = eigs(B, nev=20, which=:SM)
    ws, us, vs = Superfluids.bdg_output(d, ew, ev)
    q, ws, us, vs
end

# q, ws, us, vs = modes(d)

@load "pair_modes.jld2" q ws us vs

rvs = find_vortices(d, q)
P, _ = poles(q)
ixs = abs.(P) .> 0.5maximum(abs.(P))
clusters = cluster_adjacent(adjacent_index, keys(ixs)[ixs])
masks = [[j ∈ C for j in keys(q)] for C in clusters]

z = Superfluids.argand(d)

function fit(u, j)
    r = rvs[j]
    ixs = masks[j]
    [(z.-r)[ixs] conj((z.-r)[ixs]) ones(size(z[ixs]))] \ u[ixs]
end

function roff(q, u, v, as, j)
    qc = fit(q, j)
    uc = fit(u, j)
    vc = fit(v, j)
    rr = Complex{Float64}[]
    for α in as
        a, b, c = qc + α*uc + conj(α)*vc
        push!(rr, (b*conj(c)-conj(a)*c)/(abs2(a)-abs2(b)))
    end
    only(rr)
end

p(qs...) = plot([plot(d,q) for q in qs]...)
cosang(p,q) = abs(dot(p/norm(p), q/norm(q)))

ix = argmin(@. abs(z-rvs[1]))
ncore = abs2(ψ[ix])/d.h^2

for j = 2:3
    w = √(sum(abs2, us[j]) - sum(abs2, vs[j]))
    us[j] ./= w
    vs[j] ./= w
end

# generate core paths
hh = exp.(2π*1im*(0:0.05:1))

rv1 = [roff(q, us[2],vs[2],0.07h,1) for h in hh]
area1 = sum([imag(conj(rv1[j-1])*rv1[j]) for j in eachindex(rv1)[2:end]]) / 2

rv2 = [roff(q, us[3],vs[3],0.07h,1) for h in hh]
area2 = sum([imag(conj(rv2[j-1])*rv2[j]) for j in eachindex(rv2)[2:end]]) / 2

# use pairplots.jl
