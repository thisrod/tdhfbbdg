# Orbiting vortex plots.  See also fixed.jl

using LinearAlgebra, Plots, ComplexPhasePortrait, Colors, Optim
using Statistics: mean

using Revise
using Superfluids
using Superfluids: unroll

include("plotsty.jl")

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)
L = Superfluids.operators(s,d,:L) |> only

R = 1.5	# orbit radius
JJ = 11	# snapshots at S[JJ]
g_tol = 1e-6

ψ = steady_state(s,d)
μ = dot(ψ,L(ψ)) |> real

L, J = Superfluids.operators(s,d,:L,:J)

function wdisc(r, w, a)
    q = steady_state(s, d; rvs=[r], Ω=w, g_tol, iterations=5000, as=[a])
    w2, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
    w2-w
end

result = optimize(w->abs2(wdisc(complex(R), w, 0.5)), 0.1, 0.3, abs_tol=g_tol)

tt = range(0.0, 2π/Ω, length=16)
qq = Superfluids.integrate(s, d, q, tt; μ)

zz = find_vortex.(qq)
aa = Superfluids.unroll(@. angle(zz) - angle(zz[1]))

function imprint(θ)
    # TODO match vortex core radius
    mask = @. z-R*exp(1im*θ)
    u = @. ψ*mask/sqrt(0.25^2+abs2(mask))
    u/norm(u)
end

z = Superfluids.argand(d)
r = abs.(z)

PA = plot(d, qq[JJ], xshowaxis=false)
plot_trace(zz[aa .≤ 2π], :white)
savefig(PA, "../figs/resp200702a.pdf")

PB = plot(d, imprint(angle(zz[JJ])), xshowaxis=false, yshowaxis=false)
plot_trace(zz[aa .≤ 2π], :white)
ixs = @. 4.2-d.h<r<4.2+d.h
rr = qq[JJ][ixs]
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

PC = plot(sense_portrait(pp1,pcim) |> implot, aspect_ratio=1)
plot_trace(zz[1:JJ], :black)
savefig(PC, "../figs/resp200702c.pdf")

PD = plot(sense_portrait(pp2,pcim)  |> implot,
    aspect_ratio=1, yshowaxis=false)
plot_trace(zz[1:JJ], :black)
savefig(PD, "../figs/resp200702d.pdf")

pr = range(cl,ch, length=50)
PG = plot(pr, pr, sense_portrait(pr'), aspect_ratio=1/7, xshowaxis=false, yshowaxis=false)
ylims!(pr[1], pr[end])
savefig(PG, "../figs/resp200724c.pdf")

nin(u) = sum(abs2.(u[r .< R]))

S1t *= Ω/2π
PE = plot()
PE = scatter!(PE, S1t[1:4:end], -bphase(S1q[1:4:end])./(2π*nin(φ)); bpsty...)
# ylims!(0,5)
plot!([S1t[JJ], S1t[JJ]], [-2, 1.5]; snapsty...)
plot!(S1t, aa./2π; insty...)
scatter!(S1t[1:4:end], -bphase(imprint.(angle.(zz[1:4:end])))./(2π*nin(ψ)); impsty...)
xticks!(0:0.2:1.2)
# yticks!(0:5:30)
savefig(PE, "../figs/resp200702e.pdf")

q = slice(S1q[1])
q ./= maximum(abs, q)
y = x'

PH = plot()
plot!(PH, x[:], real(q), xlim=(-5,5), lc=:black, size=(200,100))
savefig("../figs/resp200812a.pdf")

PI = plot()
# interpolate the minimum
n0 = maximum(abs2, q)
plot!(PI, y[1:57], abs2.(q[1:57]), lc=:black, xlim=(-5,5), size=(200,100))
plot!(PI, y[58:end], lc=:black, abs2.(q[58:end]))
yy = range(y[57:58]..., length=30)
qq = range(q[57:58]..., length=30)
plot!(PI, yy, abs2.(qq), lc=:black,)
savefig("../figs/resp200812b.pdf")
