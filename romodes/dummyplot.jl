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

PA = plot(zplot(S1q[JJ]), xshowaxis=false; imopts...)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PA, "../figs/resp200702a.pdf")

PB = plot(zplot(imprint(angle(zz[JJ]))), xshowaxis=false, yshowaxis=false; imopts...)
plot_trace(zz[aa .≤ 2π], :white)
ixs = @. 4.2-h<r<4.2+h
rr = S1q[JJ][ixs]
for (θ, s) = [(0, "0"), (π/2, "pi/2"), (π, "pi"), (-π/2, "3pi/2")]
    z1 = z[ixs][argmin(@. abs(angle(rr)-θ))]
    annotate!([real(z1)], [imag(z1)], text(s, :white, 9))
end
savefig(PB, "../figs/resp200702b.pdf")

pp1 = -pci(S1q[1:JJ])/h^2
pp2 = -pci(imprint.(angle.(zz[1:JJ])))/h^2

cl = minimum(min.(pp1, pp2))
ch = maximum(max.(pp1, pp2))

@info "Color bounds" low=cl high=ch

pcim = max(-cl, ch)

PC = plot(sense_portrait(pp1,pcim) |> implot, aspect_ratio=1; imopts...)
plot_trace(zz[1:JJ], :black)
savefig(PC, "../figs/resp200702c.pdf")

PD = plot(sense_portrait(pp2,pcim)  |> implot,
    aspect_ratio=1, yshowaxis=false; imopts...)
plot_trace(zz[1:JJ], :black)
savefig(PD, "../figs/resp200702d.pdf")

pr = range(cl,ch, length=50)
PG = plot(pr, pr, sense_portrait(pr'), aspect_ratio=1/7, xshowaxis=false, yshowaxis=false,
    size=(200,55), dpi=72)
ylims!(pr[1], pr[end])
savefig(PG, "../figs/resp200724c.pdf")

nin(u) = sum(abs2.(u[r .< R]))

S1t *= Ω/2π
PE = plot()
PE = scatter!(PE, S1t[1:4:end], -bphase(S1q[1:4:end])./(2π*nin(φ)); bpsty..., sqopts...)
# ylims!(0,5)
plot!([S1t[JJ], S1t[JJ]], [ylims()[1], 1.5]; snapsty...)
plot!(S1t, aa./2π; insty...)
scatter!(S1t[1:4:end], -bphase(imprint.(angle.(zz[1:4:end])))./(2π*nin(ψ)); impsty...)
xticks!(0:0.2:1.2)
# yticks!(0:5:30)
savefig(PE, "../figs/resp200702e.pdf")

q = slice(S1q[1])
q ./= maximum(abs, q)
y = x'

PH = plot()
plot!(PH, x[:], real(q), xlim=(-5,5), lc=:black; (sqopts..., size=(200,100))...)
savefig("../figs/resp200812a.pdf")

PI = plot()
# interpolate the minimum
n0 = maximum(abs2, q)
plot!(PI, y[1:57], abs2.(q[1:57]), lc=:black, xlim=(-5,5); (sqopts..., size=(200,100))...)
plot!(PI, y[58:end], lc=:black, abs2.(q[58:end]))
yy = range(y[57:58]..., length=30)
qq = range(q[57:58]..., length=30)
plot!(PI, yy, abs2.(qq), lc=:black,)
savefig("../figs/resp200812b.pdf")
