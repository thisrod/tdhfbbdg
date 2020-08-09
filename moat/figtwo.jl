# Plot moat Berry phase figures

using JLD2

@load "moc.jld2"

include("../system.jl")
include("../figs.jl")

# fudge dynamic phase
ffs = [mean(@. Su[j]*(r>R+w)) for j = eachindex(Su)]
@. Su /= (ffs/abs(ffs))

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
    plot!(real(ins[1:j]), imag(ins[1:j]), lc=c, lw=0.5)
    plot!(real(outs[1:j]), imag(outs[1:j]), lc=c, lw=0.5)
    scat!(ins[1:1], ms=1.5)
    scat!(outs[1:1], ms=1.5)
    scat!(ins[1:9:j])
    scat!(outs[1:9:j])
end

# 72 dpi is 1pt pixels
# popts = (xlims=(-5,5), ylims=(-5,5), size=(200,200), dpi=72, leg=:none)

PA = plot(zplot(Su[j1]), xshowaxis=false; imopts...)
crumbs!(j1)
savefig(PA, "../figs/resp200716a.pdf")

PB = plot(zplot(Su[j2]), xshowaxis=false, yshowaxis=false; imopts...)
crumbs!(j2)
savefig(PB, "../figs/resp200716b.pdf")

PC = plot(zplot(Su[j3]), xshowaxis=false, yshowaxis=false; imopts...)
crumbs!(j3)
savefig(PC, "../figs/resp200716c.pdf")

PD = plot(pci(Su[1:j1]) |> sense_portrait |> implot,
    aspect_ratio=1; imopts...)
crumbs!(j1, :black)
savefig(PD, "../figs/resp200716d.pdf")
    
PE = plot(pci(Su[1:j2]) |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; imopts...)
crumbs!(j2, :black)
savefig(PE, "../figs/resp200716e.pdf")
        
PF = plot(pci(Su[1:j3]) |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; imopts...)
crumbs!(j3, :black)
savefig(PF, "../figs/resp200716f.pdf")

St *= Ω/2π
ixs = 1:12:length(St)
a = -nin(Rv)/2π*unroll(@. angle(ins[ixs])-angle(ins[1]))
b = nin(R)/2π*unroll(@. angle(outs[ixs])-angle(ins[1]))
PG = plot([St[j1], St[j1]], [-0.6, 0.2], lc=snapcol; sqopts...)
plot!([St[j3], St[j3]], [-0.6, 0.2], lc=snapcol)
plot!(St[ixs], -a; insty...)
plot!(St[ixs], -b; outsty...)
plot!(St[ixs], -a-b; nsty...)
scatter!(St[ixs], -bphase(Su)[ixs]/2π; bpsty...)
savefig(PG, "../figs/resp200724a.pdf")