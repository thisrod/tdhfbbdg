# Josephson graphs

using JLD2, Plots
using Statistics: mean
using Superfluids

R = 2.6

include("plotsty.jl")

norm2(u) = sum(abs2, u)

d = FourierDiscretisation{2}(200, 20/199)
Superfluids.default!(d)
z = argand()
r = abs.(z)

@load "moat.jld2"
tvs = St
ivs = [norm2(@. q*(r<R)) for q in Su]
ovs = [norm2(@. q*(r>R)) for q in Su]
lvs = [norm2(@. q*(r>R)*(real(z)<0)) for q in Su]
rvs = [norm2(@. q*(r>R)*(real(z)>0)) for q in Su]

plot(tvs, 1e4(lvs .- mean(extrema(lvs))), size=(200,150))
plot!(tvs, 1e4(rvs .- mean(extrema(rvs))))
savefig("../figs/resp210401a.pdf")

d = FourierDiscretisation{2}(400, 15/399)
Superfluids.default!(d)
z = argand()
r = abs.(z)

@load "mu.jld2"
tms = St
ims = [norm2(@. q*(r<R)) for q in Su]
oms = [norm2(@. q*(r>R)) for q in Su]

# Kludge to cycle through colors
plot([NaN], [NaN], size=(200,150))
plot!([NaN], [NaN])
plot!(tms, 1e4(ims .- mean(extrema(ims))))
plot!(tms, 1e4(oms .- mean(extrema(oms))))
savefig("../figs/resp210401b.pdf")