using JLD2

@load "mod.jld2"

include("../system.jl")
include("../figs.jl")

popts = (dpi = 72, leg = :none, framestyle = :box, xlims = (-5, 5), ylims = (-5, 5), size = (800, 400))

A = @animate for j = 1:length(Su)
    P = zplot(Su[j])
    xlims!(P,-5,5)
    ylims!(P,-5,5)
    Q = plot(pci(Su[1:j]) |> sense_portrait |> implot, aspect_ratio=1)
   plot(P, Q, xshowaxis=false, yshowaxis=false; popts...)
end
gif(A, "../figs/resp200812d.gif", fps=1)