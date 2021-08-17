# Effect of asymmetric density on Berry phase

using Plots
using Superfluids

include("plotsty.jl")
include("berry_utils.jl")

Superfluids.default!(:xlims, (-3,3))
Superfluids.default!(:ylims, (-2.5,3.5))

h = 15/399
d = FDDiscretisation{2}(400, h)
Superfluids.default!(d)

R = 2.0
rc = 1.0	# core radius

hh = 2π*(0:0.01:1)
uu = exp.(1im*hh)

function offtest(a, θ=0)
    z = argand()
    v = z.-a
    @. v /= abs(v)
    v[@. abs(z-a-0.5a/abs(a)*exp(1im*θ)) < 1] .= 0
    v
end

bps = Float64[]
ofs = 0:0.3:2π
for s = ofs
    bp = pci([offtest(R*u, s) for u = uu])
    push!(bps, h^2*sum(bp) / -(2π^2*R^2))
end

PA = scatter(ofs, bps)
savefig("../figs/resp210108a.pdf")

rp = R*exp(1im)
rd = rp + 0.5*exp(1.5im)
PB = plot(d, offtest(rp, 0.5))
plot!(real(R*uu), imag(R*uu), lc=:white)
scatter!(real([rp]), imag([rp]), ms=3, mc=:white, msw=0)
plot!(real([rp, rd]), imag([rp, rd]), lc=:white, arrow=true)
plot!(real([0, 2.7exp(1im)]), imag([0, 2.7exp(1im)]), lc=:white, linestyle=:dot)
savefig("../figs/resp210108b.pdf")

PC = plot(d, offtest(R*exp(1im), π/2))
plot!(real(R*uu), imag(R*uu), lc=:white)
annotate!(PC, -3.5, -3.5, "(c)", :white)

PD = plot(d, offtest(R*exp(1im), π))
plot!(real(R*uu), imag(R*uu), lc=:white)
annotate!(PD, -3.5, -3.5, "(d)", :white)

P = plot(PA, PB, PC, PD, size = (300,300))
savefig(P, "../figs/resp201218a.pdf")