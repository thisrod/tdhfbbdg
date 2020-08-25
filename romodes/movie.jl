using JLD2

N = 100
l = 20.0	# maximum domain size
C = NaN

include("../system.jl")
include("../figs.jl")

@load "acqorbit.jld2" source Ω φ ψ
@load "ss.jld2" S1t S1q

zz = [find_vortex(S1q[j]) for j = eachindex(S1q)]
R = sum(abs.(zz)) / length(zz)

function imprint(θ)
    # TODO match vortex core radius
    mask = @. z-R*exp(1im*θ)
    u = @. ψ*mask/sqrt(0.25^2+abs2(mask))
    u/norm(u)
end

function plot_trace(z, col)
    plot!(real.(z), imag.(z), lc=col, leg=:none)
    scatter!(real.(z[1:1]), imag.(z[1:1]), ms=5, mc=col, msw=0)
end

popts = (dpi = 72, leg = :none, framestyle = :box, xlims = (-5, 5), ylims = (-5, 5), size = (800, 400))

A = @animate for j = 1:length(S1t)
    P = zplot(S1q[j])
    plot_trace(zz[1:j], :white)
    xlims!(P,-5,5)
    ylims!(P,-5,5)
    Q = plot(pci(S1q[1:j]) |> sense_portrait |> implot, aspect_ratio=1)
    plot_trace(zz[1:j], :black)
   plot(P, Q, xshowaxis=false, yshowaxis=false; popts...)
end
gif(A, "../figs/resp200811a.gif", fps=3)

S = imprint.(angle.(zz))

B = @animate for j = 1:length(S1t)
    P = zplot(S[j])
    plot_trace(zz[1:j], :white)
    xlims!(P,-5,5)
    ylims!(P,-5,5)
    Q = plot(pci(S[1:j]) |> sense_portrait |> implot, aspect_ratio=1)
    plot_trace(zz[1:j], :black)
   plot(P, Q, xshowaxis=false, yshowaxis=false; popts...)
end
gif(B, "../figs/resp200819a.gif", fps=3)