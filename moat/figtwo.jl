# Plot moat Berry phase figures

using JLD2

@load "mob.jld2"

include("../system.jl")
include("../figs.jl")

ψ = Su[1]	# TODO relax density without a vortex

# fudge dynamic phase
ffs = [mean(@. Su[j]*(r>R+w)) for j = eachindex(Su)]
@. Su /= (ffs/abs(ffs))

j1 = 7
j2 = 18
j3 = 35

ins = [find_vortex(q) for q in Su]
outs = [find_moat(q) for q in Su]
Rv = mean(abs.(ins))

nin(R) = sum(abs2.(ψ[r .< R]))

function crumbs!(j)
    plot!(ins[1:j], lc=:white, lw=0.5)
    plot!(outs[1:j], lc=:white, lw=0.5)
    scatter!(ins[1:1], ms=1.5, mc=:white, msw=0)
    scatter!(outs[1:1], ms=1.5, mc=:white, msw=0)
    scatter!(ins[4:3:j], ms=1, mc=:white, msw=0)
    scatter!(outs[4:3:j], ms=1, mc=:white, msw=0)
end

# 72 dpi is 1pt pixels
popts = (xlims=(-7,7), ylims=(-7,7), size=(200,200), dpi=72, leg=:none,
    xlabel="", ylabel="")

PA = plot(zplot(Su[j1]), xshowaxis=false; popts...)
crumbs!(j1)
savefig(PA, "../figs/resp200716a.pdf")

PB = plot(zplot(Su[j2]), xshowaxis=false, yshowaxis=false; popts...)
crumbs!(j2)
savefig(PB, "../figs/resp200716b.pdf")

PC = plot(zplot(Su[j3]), xshowaxis=false, yshowaxis=false; popts...)
crumbs!(j3)
savefig(PC, "../figs/resp200716c.pdf")

PD = plot(pci(Su[1:j1]) |> sense_portrait |> implot,
    aspect_ratio=1; popts...)
savefig(PD, "../figs/resp200716d.pdf")
    
PE = plot(pci(Su[1:j2]) |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; popts...)
savefig(PE, "../figs/resp200716e.pdf")
        
PF = plot(pci(Su[1:j3]) |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; popts...)
savefig(PF, "../figs/resp200716f.pdf")

St *= Ω/2π
ixs = 1:4:length(St)
a = -nin(Rv)/2π*unroll(@. angle(ins[ixs])-angle(ins[1]))
b = nin(R)/2π*unroll(@. angle(outs[ixs])-angle(ins[1]))
PG = scatter(St[ixs], bphase(Su)[ixs]/2π, label="GPE Berry",
    leg=:topleft, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=3,
    size=(200,200), dpi=72)
scatter!(St[ixs], a, label="N_in", ms=3)
scatter!(St[ixs], b, label="N_out", ms=3)
scatter!(St[ixs], a+b, label="N", ms=3)
savefig(PG, "../figs/resp200724a.pdf")