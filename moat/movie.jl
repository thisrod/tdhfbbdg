using JLD2

@load "moc.jld2"

include("../system.jl")
include("../figs.jl")

# fudge dynamic phase
ffs = [mean(@. Su[j]*(r>R+w)) for j = eachindex(Su)]
@. Su /= (ffs/abs(ffs))

ins = [find_vortex(q) for q in Su]
outs = [find_moat(q) for q in Su]

function plot_trace(z, col)
    plot!(real.(z), imag.(z), lc=col, leg=:none)
    scatter!(real.(z[1:1]), imag.(z[1:1]), ms=5, mc=col, msw=0)
end

popts = (dpi = 72, leg = :none, framestyle = :box, xlims = (-5, 5), ylims = (-5, 5), size = (800, 400))

A = @animate for j = 1:2:length(St)
    P = zplot(Su[j])
    plot_trace(ins[1:j], :white)
    plot_trace(outs[1:j], :white)
    xlims!(P,-5,5)
    ylims!(P,-5,5)
    Q = plot(pci(Su[1:j]) |> sense_portrait |> implot, aspect_ratio=1)
    plot_trace(ins[1:j], :black)
    plot_trace(outs[1:j], :black)
   plot(P, Q, xshowaxis=false, yshowaxis=false; popts...)
end
gif(A, "../figs/resp200811b.gif", fps=3)