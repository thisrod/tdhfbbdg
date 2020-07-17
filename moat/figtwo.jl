# Plot moat Berry phase figures

using JLD2

@load "moa.jld2"

include("../system.jl")
include("../figs.jl")

j1 = 7
j2 = 18
j3 = 35

# 72 dpi is 1pt pixels
popts = (xlims=(-7,7), ylims=(-7,7), size=(200,200), dpi=72)

PA = plot(zplot(Su[j1]), xshowaxis=false; popts...)
savefig(PA, "../figs/resp200716a.pdf")

PB = plot(zplot(Su[j2]), xshowaxis=false, yshowaxis=false; popts...)
savefig(PB, "../figs/resp200716b.pdf")

PC = plot(zplot(Su[j3]), xshowaxis=false, yshowaxis=false; popts...)
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
