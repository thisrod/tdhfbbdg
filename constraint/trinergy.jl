# Energy landscape for an orbiting vortex and 3-lattice

using LinearAlgebra, Plots, Optim

using Revise
using Superfluids

default(:legend, :none)

s = Superfluid{2}(3000, (x,y)->x^2+y^2)
Superfluids.default!(s)
d = FDDiscretisation(100, 20)
Superfluids.default!(d)
Superfluids.default!(:g_tol, 1e-6)

z = argand()
L, H = Superfluids.operators(:L, :H)

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
R_TF = √μ

rr = 0:0.2:R_TF
Ωs = 0:0.1:0.6

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

# compare energy to vortex-free steady state at Ω
oqs = Array{Any}(undef,length(rr),length(Ωs))
oEs = similar(oqs, Float64)
rdls = similar(oqs, Float64)
for j = eachindex(Ωs)
    ψ = steady_state(Ω=Ωs[j])
    E = dot(H(ψ, Ωs[j]), ψ) |> real
    for k = eachindex(rr)
        q = Superfluids.relax_field(s, d, [Complex(rr[k])], Ωs[j])
        oqs[k,j] = q
        oEs[k,j] = real(dot(H(q, Ωs[j]), q)) - E
        rdls[k,j] = rsdl(q, Ωs[j])
    end
end

P = plot()
for j = 1:length(Ωs)
    scatter!(rr, oEs[:,j], label="$(Ωs[j])", leg=:topright)
end

rr = 0.5:0.2:R_TF
uu = exp.(2π*1im*(0:2)/3)
tqs = [Superfluids.relax_field(s, d, r*uu, Ω) for r in rr[2:end]]
tEs = [dot(H(q, Ω), q) |> real for q in tqs]
rdls = [rsdl(q, Ω) for q in tqs]

# plot(scatter(rr, oEs), scatter(rr, rdls), layout=@layout [a;b])