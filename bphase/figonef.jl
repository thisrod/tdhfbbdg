# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, JLD2
using Superfluids

Plots.default(:legend, :none)

@load "fixed.jld2"

input("plotsty.jl")

if false

PF = plot()
plot!([R/R_TF, R/R_TF], [0.0, 0.4]; snapsty...)
plot!(rin/R_TF, nin; insty...)
scatter!(ss/R_TF, ip; impsty...)
scatter!(ss/R_TF, gp; bpsty...)
xlims!(0,1)
xticks!([0, 0.5, 1])
savefig(PF, "resp200812e.pdf")

# plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])

end