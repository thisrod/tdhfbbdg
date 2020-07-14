# Process ground state and dynamics data

using Plots, ComplexPhasePortrait, JLD2, LinearAlgebra
using Statistics: mean

# Plots and portraits are a bit wierd with pixel arrays
implot(x,y,image) = plot(x, y, image,
    yflip=false, aspect_ratio=1, framestyle=:box, tick_direction=:out)
implot(image) = implot(y,y,image)
saneportrait(u) = reverse(portrait(u), dims=1)
zplot(u) = implot(saneportrait(u).*abs2.(u)/maximum(abs2,u))
argplot(u) = implot(saneportrait(u))
zplot(u::Matrix{<:Real}) = zplot(u .|> Complex)
argplot(u::Matrix{<:Real}) = argplot(u .|> Complex)

N = 100
l = 20.0	# maximum domain size
J = 11	# snapshots at S[J]

h = min(l/(N+1), sqrt(√2*π/N))
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)

y = [-10, 10]	# Plots.jl pixel offset

@load "acqorbit.jld2" source Ω φ ψ
@load "ss.jld2" S1t S1q

include("../figs.jl")

zz = [find_vortex(S1q[j], 2.5) for j = eachindex(S1q)]
R = mean(abs.(zz))
aa = unroll(@. angle(zz) - angle(zz[1]))

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

# 72 dpi is 1pt pixels
popts = (xlims=(-5,5), ylims=(-5,5), size=(200,200), dpi=72)

PA = plot(zplot(S1q[J]), xshowaxis=false; popts...)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PA, "../figs/resp200702a.pdf")

PB = plot(zplot(imprint(angle(zz[J]))), xshowaxis=false, yshowaxis=false; popts...)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PB, "../figs/resp200702b.pdf")

PC = plot(pci(S1q[1:J]) |> sense_portrait |> implot,
    aspect_ratio=1; popts...)
plot_trace(zz[1:J], :black)
savefig(PC, "../figs/resp200702c.pdf")

PD = plot(-pci(imprint.(angle.(zz[1:J]))) |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; popts...)
plot_trace(zz[1:J], :black)
savefig(PD, "../figs/resp200702d.pdf")

nin(u) = sum(abs2.(u[r .< R]))

S1t *= Ω/2π
PE = scatter(S1t[1:4:end], bphase(S1q[1:4:end])./(2π*nin(φ)), label="GPE Berry",
    leg=:none, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=3,
    size=(200,200), dpi=72)
plot!([S1t[J], S1t[J]], [ylims()[1], 1.5], lc=RGB(0.3,0,0), label="snapshots")
plot!(S1t, aa./2π, lw=2, label="Wu")
scatter!(S1t[1:4:end], -bphase(imprint.(angle.(zz[1:4:end])))./(2π*nin(ψ)), label="Imp. Berry", ms=3)
xticks!(0:0.2:1.2)
# yticks!(0:5:30)
savefig(PE, "../figs/resp200702e.pdf")

run(`xetex figone.tex`)