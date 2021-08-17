# Orbiting vortex plots.  See also fixed.jl

using LinearAlgebra, Plots, ComplexPhasePortrait, Colors, JLD2, Interpolations
using Statistics: mean

using Revise
using Superfluids
using Superfluids: unroll, sense_portrait, saneportrait

include("plotsty.jl")
include("berry_utils.jl")

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)
L = Superfluids.operators(s,d,:L) |> only

R = 1.5	# orbit radius
JJ = 11	# snapshots at S[JJ]
g_tol = 1e-6

ψ = steady_state(s,d)
μ = dot(ψ,L(ψ)) |> real

x, y = Superfluids.daxes(d)

nin =
    let
        y = d.h/2*(1-d.n:2:d.n-1)
        # CSI only takes a range, not an array
        f = CubicSplineInterpolation((y, y), ψ)
        yy = range(y[1], y[end]; length=5*d.n)
        rs = hypot.(yy', yy)
        φ = f.(yy', yy) |> normalize
        sum(@. abs2(φ)*(rs<R))
    end

@load "orbit.jld2"

function imprint_phase(u)
    r₀ = find_vortex(d,u)
    @. abs(u)*(z-r₀)/abs(z-r₀)
end

zz = [find_vortex(d, q) for q in qq]
aa = unroll(@. angle(zz) - angle(zz[1]))

z = Superfluids.argand(d)
r = abs.(z)

θ = 2π/3
q = qq[JJ]
qi = imprint_phase(q)
rv = find_vortex(d, q)

ixs = @. abs(angle(q*exp(-1im*θ))) < d.h/2abs(z-rv)
img = saneportrait(q)
img[ixs] .= colorant"gray"
img .*= abs2.(q) / maximum(abs2, q)
PA = plot(x[:], y[:], transpose(img), xshowaxis=false, yshowaxis=false,
    yflip=false,
    aspect_ratio=1,
    tick_direction=:out,
    xlims=(-5,5),
    ylims=(-5,5)
)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PA, "../figs/resp200702a.pdf")

ixs = @. abs(angle(qi*exp(-1im*θ))) < d.h/2abs(z-rv)
PB = plot(d, qi, xshowaxis=false)
plot_trace(zz[aa .≤ 2π], :white)
scatter!(real(z[ixs]), imag(z[ixs]), mc=:gray, ms=1, msw=0)
jxs = @. 4.2-d.h<r<4.2+d.h
rr = qi[jxs]
for (θ, s) = [(0, "0"), (π/2, "pi/2"), (π, "pi"), (-π/2, "3pi/2")]
    z1 = z[jxs][argmin(@. abs(angle(rr)-θ))]
    annotate!([real(z1)], [imag(z1)], text(s, :white, 9))
end
savefig(PB, "../figs/resp200702b.pdf")

pp1 = -pci(qq[1:JJ])/d.h^2
pp2 = -pci(imprint_phase.(qq[1:JJ]))/d.h^2

cl = minimum(min.(pp1, pp2))
ch = maximum(max.(pp1, pp2))

@info "Color bounds" low=cl high=ch

pcim = max(-cl, ch)

Superfluids.default!(:clim, pcim)
PC = plot(d, pp1, aspect_ratio=1, yshowaxis=false)
plot_trace(zz[1:JJ], :black, true)
savefig(PC, "../figs/resp200702c.pdf")

PD = plot(d, pp2, aspect_ratio=1)
plot_trace(zz[1:JJ], :black, true)
savefig(PD, "../figs/resp200702d.pdf")

pr = range(cl,ch, length=50)
PG = plot(pr, [0], sense_portrait(pr'), xshowaxis=false, yshowaxis=false,
    size=(200,55), aspect_ratio=:none)
savefig(PG, "../figs/resp200724c.pdf")

tt *= Ω/2π
PE = plot()
PE = scatter!(PE, tt[1:4:end], -bphase(qq)[1:4:end]./(2π*nin); bpsty...)
# ylims!(0,5)
plot!([tt[JJ], tt[JJ]], [-2, 1.5]; snapsty...)
plot!(tt, aa./2π; insty...)
scatter!(tt[1:4:end], -bphase(imprint_phase.(qq))[1:4:end]./(2π*nin); impsty...)
xticks!(0:0.2:1.2)
# yticks!(0:5:30)
savefig(PE, "../figs/resp200702e.pdf")

q = slice(qq[1])/d.h^2

PH = plot(x[:], real(q), xlim=(-5,5), lc=:black, size=(200,100))
savefig("../figs/resp200812a.pdf")

PI = plot(x[:], abs2.(q), xlim=(-5,5), lc=:black, size=(200,100))
savefig("../figs/resp200812b.pdf")

Superfluids.default!(:clim, nothing)
Plots.default(:size, (600,300))
Plots.default(:dpi, 3*Plots.default(:dpi))

PJ = @animate for j = 2:length(qq)
    plot(plot(d,qq[j]), plot(d, -pci(qq[1:j])))
end
gif(PJ, "../figs/SV2.gif", fps=3)

PK = @animate for j = 2:length(qq)
    plot(plot(d,imprint_phase(qq[j])), plot(d, -pci(imprint_phase.(qq[1:j]))))
end
gif(PK, "../figs/SV1.gif", fps=3)
