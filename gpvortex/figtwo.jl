# Plot moat Berry phase figures

using JLD2, Plots
using Superfluids
using Superfluids: unroll, sense_portrait
using Statistics: mean

R = 2.6
w = 0.2

include("plotsty.jl")
include("berry_utils.jl")

Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))

d = FourierDiscretisation{2}(200, 20/199)
Superfluids.default!(d)
z = argand()
r = abs.(z)

hh = 2π*(0:0.01:1)
uu = exp.(1im*hh)

@load "moat.jld2"

function find_moat(u)
    P, Q = Superfluids.poles(u)
    v = @. (R-w/2 < r < R+w/2)*abs(P+conj(Q))/abs(u)
    ixs = v .> 0.5maximum(v)
    find_vortex(d, u, ixs)
end

ψ = Su[1]	# TODO relax density without a vortex

j1 = 3*7
j2 = 3*18
j3 = 3*35

ins = [find_vortex(q) for q in Su]
outs = [find_moat(q) for q in Su]
Rv = mean(abs.(ins))

nin(R) = sum(abs2.(ψ[r .< R]))


function crumbs!(j, c=:white)
    scat!(zz; ms=1) = scatter!(real(zz), imag(zz), ms=ms, mc=c, msw=0)
    plot!(real(ins[1:j]), imag(ins[1:j]), lc=c, arrow=true)
    plot!(real(outs[1:j]), imag(outs[1:j]), lc=c, ls=:dot, arrow=true)
    scat!(ins[1:1], ms=1.5)
    scat!(outs[1:1], ms=1.5)
    scat!(ins[1:9:j])
    scat!(outs[1:9:j])
end

PA = plot(d, Su[j1], xshowaxis=false)
crumbs!(j1)
savefig(PA, "../figs/resp200716a.pdf")

PB = plot(d, Su[j2], xshowaxis=false, yshowaxis=false)
crumbs!(j2)
savefig(PB, "../figs/resp200716b.pdf")

PC = plot(d, Su[j3], xshowaxis=false, yshowaxis=false)
crumbs!(j3)
savefig(PC, "../figs/resp200716c.pdf")

pp1 = -pci(Su[1:j1])/d.h^2
pp3 = -pci(Su[1:j3])/d.h^2

cl = -0.148
ch = 0.083

@info "Color bounds" low=cl high=ch

Superfluids.default!(:clim, 0.15)
PD = plot(d, pp1)
crumbs!(j1, :black)
savefig(PD, "../figs/resp200716d.pdf")
    
PE = plot(d, pci(Su[1:j2]), yshowaxis=false)
crumbs!(j2, :black)
savefig(PE, "../figs/resp200716e.pdf")
        
PF = plot(d, pp3, yshowaxis=false)
crumbs!(j3, :black)
savefig(PF, "../figs/resp200716f.pdf")

pr = range(cl,ch, length=50)
PH = plot(pr, [0], sense_portrait(pr'), xshowaxis=false, yshowaxis=false,
    size=(200,55), aspect_ratio=:none)
savefig(PH, "../figs/resp200812c.pdf")

St *= Ω/2π
ixs = 1:12:length(St)
a = -nin(Rv)/2π*unroll(@. angle(ins[ixs])-angle(ins[1]))
b = nin(R)/2π*unroll(@. angle(outs[ixs])-angle(ins[1]))
PG = plot([St[j1], St[j1]], [-0.4, 0.1], size=(200,300); snapsty...)
plot!([St[j3], St[j3]], [-0.4, 0.1]; snapsty...)
plot!(St[ixs], -a; insty...)
plot!(St[ixs], -b, ls=:dot; outsty...)
plot!(St[ixs], -a-b; nsty...)
scatter!(St[ixs], -bphase(Su)[ixs]/2π; bpsty...)
xlims!(0,1)
savefig(PG, "../figs/resp200724a.pdf")

# ψ is real along the diameter through the vortices
PI = plot(Superfluids.daxes(d)[1], slice(Su[1])/d.h^2 |> real, xlim=(-5,5), lc=:black, size=(200,100))
savefig(PI, "../figs/resp201219a.pdf")

Superfluids.default!(:clim, nothing)
Plots.default(:size, (600,300))
Plots.default(:dpi, 3*Plots.default(:dpi))

# TODO q in SU
PJ = @animate for j = 2:length(Su)
    plot(plot(d,Su[j]), plot(d, -pci(Su[1:j])))
end
gif(PJ, "../figs/SV3.gif", fps=3)

q = Su[j1]
nmax = maximum(abs2, @. q*(r>R))
mask = @. abs2(q) > nmax/2
r1 = maximum(r[@. mask&(r<R)])
r2 = minimum(r[@. mask&(r>R)])
PK = plot(d, (@. Su[j1]/abs(Su[j1])))
scatter!([real(ins[j1])], [imag(ins[j1])], mc=:white, ms=2.5)
scatter!([real(outs[j1])], [imag(outs[j1])], mc=:white, ms=2.5)
plot!(r1*sin.(hh), r1*cos.(hh), lc=:black)
plot!(r2*sin.(hh), r2*cos.(hh), lc=:black)
xlims!(-0.5, 2.5)
ylims!(0,3)
savefig("../figs/resp210106a.pdf")