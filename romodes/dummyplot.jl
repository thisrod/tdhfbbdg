# Process ground state and dynamics data

using JLD2, LinearAlgebra
using Statistics: mean

N = 100
l = 20.0	# maximum domain size
C = NaN
JJ = 11	# snapshots at S[JJ]

include("../system.jl")

y = [-10, 10]	# Plots.jl pixel offset

@load "acqorbit.jld2" source Ω φ ψ
@load "ss.jld2" S1t S1q

include("../figs.jl")

zz = [find_vortex(S1q[j]) for j = eachindex(S1q)]
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

PA = plot(zplot(S1q[JJ]), xshowaxis=false; popts...)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PA, "../figs/resp200702a.pdf")

PB = plot(zplot(imprint(angle(zz[JJ]))), xshowaxis=false, yshowaxis=false; popts...)
plot_trace(zz[aa .≤ 2π], :white)
ixs = @. 4.2-h<r<4.2+h
rr = S1q[JJ][ixs]
for (θ, s) = [(0, "0"), (π/2, "pi/2"), (π, "pi"), (-π/2, "3pi/2")]
    z1 = z[ixs][argmin(@. abs(angle(rr)-θ))]
    annotate!([real(z1)], [imag(z1)], text(s, :white, 9))
end
savefig(PB, "../figs/resp200702b.pdf")

pp1 = pci(S1q[1:JJ])/h^2
PC = plot(pp1 |> sense_portrait |> implot, aspect_ratio=1; popts...)
plot_trace(zz[1:JJ], :black)
savefig(PC, "../figs/resp200702c.pdf")

pp2 = -pci(imprint.(angle.(zz[1:JJ])))/h^2
PD = plot(pp2 |> sense_portrait |> implot,
    aspect_ratio=1, yshowaxis=false; popts...)
plot_trace(zz[1:JJ], :black)
savefig(PD, "../figs/resp200702d.pdf")

pr = range(minimum(pp1), maximum(pp1), length=50)
pr = reshape(pr,1,:)
PF = plot(pr[:], pr[:], sense_portrait(pr), aspect_ratio=1/7, yshowaxis=false,
    framestyle=:box, tick_direction=:out, size=(200,55), dpi=72)
ylims!(minimum(pp1), maximum(pp1))
xticks!([-0.03, 0, 0.03, 0.06])
savefig(PF, "../figs/resp200724b.pdf")

pr = range(minimum(pp2), maximum(pp2), length=50)
pr = reshape(pr,1,:)
PG = plot(pr[:], pr[:], sense_portrait(pr), aspect_ratio=1/7, yshowaxis=false,
    framestyle=:box, tick_direction=:out, size=(200,55), dpi=72)
ylims!(minimum(pp2), maximum(pp2))
xticks!([-0.03, 0, 0.03, 0.06])
savefig(PG, "../figs/resp200724c.pdf")

nin(u) = sum(abs2.(u[r .< R]))

S1t *= Ω/2π
PE = scatter(S1t[1:4:end], bphase(S1q[1:4:end])./(2π*nin(φ)), label="GPE Berry",
    leg=:none, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=3,
    size=(200,200), dpi=72)
plot!([S1t[JJ], S1t[JJ]], [ylims()[1], 1.5], lc=RGB(0.3,0,0), label="snapshots")
plot!(S1t, aa./2π, lw=2, label="Wu")
scatter!(S1t[1:4:end], -bphase(imprint.(angle.(zz[1:4:end])))./(2π*nin(ψ)), label="Imp. Berry", ms=3)
xticks!(0:0.2:1.2)
# yticks!(0:5:30)
savefig(PE, "../figs/resp200702e.pdf")
