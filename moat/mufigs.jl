# Moat potential

using DifferentialEquations, JLD2

@load "mu.jld2"

include("../system.jl")
include("../figs.jl")

# Add moat
@. V += 100*exp(-(r-R)^2/2/w^2)

# 72 dpi is 1pt pixels
popts = (xlims=(-5,5), ylims=(-5,5), size=(200,200), dpi=72, leg=:none)

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

function derplot(q)
    P1 = scatter(r[mx], angle.(q[mx]))
    xlims!(R+hh[1], R+hh[end])
    dd = [Df(q, R+a) for a in hh]
    P2 = scatter(R.+hh, dd)
    P1,P2
end

PA = plot(Su[end] |> zplot; imopts...)
savefig("../figs/resp200805a.pdf")

PB = plot(pci(Su) |> sense_portrait |> implot,
    aspect_ratio=1; imopts...)
savefig("../figs/resp200805b.pdf")

PC, PD = derplot(Su[end])

PC = plot(PC; ms=2, msw=0, mc=:black, xticks=[2.4, 2.6, 2.8], recopts...)
savefig("../figs/resp200805c.pdf")

PD = plot(PD; ms=2, msw=0, mc=:black, xticks=[2.4, 2.6, 2.8], recopts...)
savefig("../figs/resp200805d.pdf")

nin = sum(@. abs2(Su[1])*(r<R))
PE = plot(St, nin*μoff.*St; insty..., sqopts...)
scatter!(St, bphase(Su); bpsty...)
savefig("../figs/resp200805e.pdf")
