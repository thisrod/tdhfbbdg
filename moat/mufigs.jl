# Moat potential

using DifferentialEquations, JLD2

@load "mod.jld2"

jj = 9	# snapshot index

include("../system.jl")
include("../figs.jl")

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

mx = @. (R-w) < r < (R+w)
rr = r[mx]
hh = -0.2:0.0002:0.2
function Df(q,s)
    ff = angle.(q[mx])
    Dfs = Float64[]
    ix = sortperm(@. abs(rr-s))[1:250]
    cs = (@. ((rr[ix]-s))^(0:2)') \ ff[ix]
    cs[2]
end

function derplot(qs...)
    P1 = plot()
    P2 = plot()
    xlims!(R+hh[1], R+hh[end])
    for q in qs
        scatter!(P1, r[mx], angle.(q[mx]))
        scatter!(P2, R.+hh, [Df(q, R+a) for a in hh])
    end
    P1,P2
end

PA = plot(Su[jj] |> zplot; imopts...)
savefig("../figs/resp200805a.pdf")

PB = plot(pci(Su[1:jj]) |> sense_portrait |> implot,
    aspect_ratio=1; imopts...)
savefig("../figs/resp200805b.pdf")

PC, PD = derplot(Su[jj])

PC = plot(PC; ms=2, msw=0, mc=:green, xticks=[2.4, 2.6, 2.8], recopts...)
savefig("../figs/resp200805c.pdf")

PD = plot(PD; ms=2, msw=0, mc=:green, xticks=[2.4, 2.6, 2.8], recopts...)
savefig("../figs/resp200805d.pdf")

nin = sum(@. abs2(Su[1])*(r<R))
PE = plot([St[jj], St[jj]]/2π, [0.15, 0.45], lc=RGB(0.3,0,0), label="snapshots")
plot!(St/2π, nin*μoff.*St/2π; insty..., sqopts...)
scatter!(St/2π, bphase(Su)/2π; bpsty...)
savefig("../figs/resp200805e.pdf")


PF = plot(; leg=:none)
for j = 2:10
    scatter!(Su[j][mx], label="$(St[j])", ms=1.5, msw=0)
end