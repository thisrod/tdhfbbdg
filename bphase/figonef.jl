# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, JLD2
using Superfluids

Plots.default(:legend, :none)

@load "fixed.jld2"

# Colors for Berry phase plots
snapsty = RGB(0.3,0,0)
bpsty, impsty, insty, outsty, nsty =
    distinguishable_colors(5+3, [snapsty, RGB(1,1,1), RGB(0,0,0)])[4:end]
bpsty = (ms=2, mc=bpsty, msc=0.5bpsty)
impsty = (ms=2, mc=impsty, msc=0.5impsty)
insty = (lc=insty,)
outsty = (lc=outsty,)
nsty = (lc=nsty,)
snapsty = (lc=snapsty, lw=0.5)

popts = (dpi=72, leg=:none, framestyle=:box, fontfamily="Latin Modern Sans")
imopts = (popts..., xlims=(-5,5), ylims=(-5,5), size=(200,200))
sqopts = (popts..., size=(200,200))
recopts = (popts..., size=(100,200))

if false

PF = plot(; sqopts...)
plot!([R/R_TF, R/R_TF], [0.0, 0.4]; snapsty...)
plot!(rin/R_TF, nin; insty...)
scatter!(ss/R_TF, ip; impsty...)
scatter!(ss/R_TF, gp; bpsty...)
xlims!(0,1)
xticks!([0, 0.5, 1])
savefig(PF, "resp200812e.pdf")

# plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])

end