# Moat potential

using DifferentialEquations, JLD2, Plots
using Superfluids

@load "mu.jld2"

include("plotsty.jl")
include("berry_utils.jl")

d = FourierDiscretisation{2}(400, 15/399)
Superfluids.default!(d)
Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))

n₀ = maximum(abs2, Su[1])/d.h^2

function sce(u, hoff=0)
   u = slice(u)
   P1 = scatter(y, abs2.(u)/h^2, mc=:black, ms=1.5, msw=0, leg=:none)
   ylabel!("|psi|^2")
   P2 = scatter(y, unroll(angle.(u)) .+ 2π*hoff, mc=:green, ms=1.5, msw=0,  leg=:none)
   ylabel!("arg(psi)")
   xlabel!("x")
   plot(P1, P2, layout=@layout [a;b])
   P1, P2, plot(P1, P2, layout=@layout [a;b])
end

r = sample(hypot)
mx = @. (R-w) < r < (R+w)
rr = r[mx]
rix = sortperm(rr)
hh = -0.2:0.0002:0.2
function Df(q,s)
    ff = angle.(q[mx])
    Dfs = Float64[]
    ix = sortperm(@. abs(rr-s))[1:250]
    cs = (@. ((rr[ix]-s))^(0:2)') \ ff[ix]
    cs[2]
end

function derplot(qs...)
    P1 = plot(ms=2, msw=0, xticks=[2.4, 2.6, 2.8], size=(100,200))
    P2 = plot(ms=2, msw=0, xticks=[2.4, 2.6, 2.8], size=(100,200))
    xlims!(R+hh[1], R+hh[end])
    for q in qs
        plot!(P1, rr[rix], angle.(q[mx])[rix])
        plot!(P2, R.+hh, [Df(q, R+a) for a in hh]/1000/n₀)
    end
    P1,P2
end

jj = 7	# snapshot index

PA = plot(d, Su[10])
savefig("../figs/resp200805a.pdf")

Superfluids.default!(:clim, 0.15)
PB = plot(d, pci(Su[1:jj+1])/d.h^2)
savefig("../figs/resp200805b.pdf")

PC, PD = derplot(Su[jj+1:-1:jj-1]...)

savefig(PC, "../figs/resp200805c.pdf")

savefig(PD, "../figs/resp200805d.pdf")

nin = sum(@. abs2(Su[1])*(r<R))
PE = plot()
for j = jj-1:jj+1
    plot!([St[j], St[j]]/2π, [0.2, 0.5]; snapsty...)
end
plot!(St/2π, μoff.*St/2π; insty...,)
scatter!(St/2π, bphase(Su)/2π/nin; bpsty...)
savefig("../figs/resp200805e.pdf")

q = Su[1]
nmax = maximum(abs2, @. q*(r>R))
mask = @. abs2(q) > nmax/2
r1 = maximum(r[@. mask&(r<R)])
r2 = minimum(r[@. mask&(r>R)])
PF = plot(Superfluids.daxes(d)[1], real(slice(q))/d.h^2, xlim=(-5,5), lc=:black, size=(200,100))
plot!([r1, r1], collect(ylims()), ls=:dot, lc=:black)
plot!([r2, r2], collect(ylims()), ls=:dot, lc=:black)
savefig(PF, "../figs/resp201219c.pdf")

plot(; leg=:none)
for j = 2:10
    scatter!(Su[j][mx], label="$(St[j])", ms=1.5, msw=0)
end

Superfluids.default!(:clim, nothing)
Plots.default(:dpi, 3*Plots.default(:dpi))

PG = @animate for j = 2:length(Su)
    P, Q = derplot(Su[j])
    ylims!(P, (-π,π))
    ylims!(Q, (-3.5, 3.5))
    plot(plot(d,Su[j]), plot(d, pci(Su[1:j])), P, Q, size=(600, 600))
end
gif(PG, "../figs/SV4.gif", fps=2)
