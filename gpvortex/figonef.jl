# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, JLD2, Interpolations
using Superfluids

Plots.default(:legend, :none)

@load "fixed.jld2"

include("plotsty.jl")
include("berry_utils.jl")

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)
R = 1.5	# orbit radius

z = Superfluids.argand(d)
r = abs.(z)

L, J = Superfluids.operators(s,d, :L, :J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(2μ)

# interpolate nin
y = d.h/2*(1-d.n:2:d.n-1)
f = CubicSplineInterpolation((y, y), ψ)
yy = range(y[1], y[end]; length=5*d.n)
rs = hypot.(yy', yy)
ψ = f.(yy', yy)
ψ ./= norm(ψ)
rin = 0:d.h/5:R_TF
nin = [sum(@. abs2(ψ)*(rs<s)) for s=rin]

function imprint_phase(u)
    r₀ = find_vortex(d,u)
    @. abs(u)*(z-r₀)/abs(z-r₀)
end

r1s = [find_vortex(d, qs[1]) for qs in S]
r2s = [find_vortex(d, qs[end]) for qs in S]
hs = @. angle(r2s*conj(r1s))
gp = [x[end] for x in bphase.(S)]./hs
ip = [x[end] for x in bphase.(imprint_phase.(s) for s in S)]./hs

PF = plot()
plot!([R/R_TF, R/R_TF], [0.0, 0.4]; snapsty...)
plot!(rin/R_TF, nin; insty...)
scatter!(rr/R_TF, -ip; impsty...)
scatter!(rr/R_TF, -gp; bpsty...)
xlims!(0,1)
xticks!([0, 0.5, 1])
savefig(PF, "../figs/resp200812e.pdf")

# plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])

Plots.default(:dpi, 3*Plots.default(:dpi))
Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))

PG = @animate for qs in S
    plot(plot(d, qs[1]), plot(d, @. qs[1]/abs(qs[1])), size=(600,300))
end
gif(PG, "../figs/SV5.gif", fps=1)
