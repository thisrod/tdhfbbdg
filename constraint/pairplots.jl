using LinearAlgebra, Plots, JLD2

using Revise
using Superfluids

default(:legend, :none)

Superfluids.default!(Superfluid{2}(3000, (x,y)->x^2+y^2))
Superfluids.default!(FDDiscretisation(150, 20))
Superfluids.default!(:g_tol, 1e-7)
g_tol = 1e-7
s = Superfluids.default(:superfluid)
d = Discretisation()

@load "pair_modes.jld2"

N = d.n

us = [reshape(uvs[1:N^2, j], N, N) for j = axes(uvs,2)]
vs = [reshape(uvs[N^2+1:end, j], N, N) for j = axes(uvs,2)]

J = Superfluids.operators(:J)[]
jj = [dot(J(us[k]), us[k]) |> real for k = eachindex(ws)]

PA = plot(d, q)
savefig("../figs/resp200828a.pdf")

PB = scatter(jj, ws)
scatter!(jj[2:2], ws[2:2], mc=:yellow, msc=:royalblue, msw=2)
scatter!(jj[3:3], ws[3:3], mc=:lightgreen)
xlabel!("J for u mode")
ylabel!("w (uncertain units)")
savefig("../figs/resp200828b.pdf")

PC = plot(plot(d, us[1]), plot(d, vs[1]))
title!("zero mode")
savefig("../figs/resp200828c.pdf")

PD = plot(plot(d, us[2]), plot(d, vs[2]))
title!("yellow/blue KT")
savefig("../figs/resp200828d.pdf")

PE = plot(p(umode(3)), p(vmode(3)))
title!("green KT")
savefig("../figs/resp200828e.pdf")

uu = exp.(2π*1im*(0:0.05:1))

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
function pci(S)
    if length(S) == 1
        zero(S[])
    else
        [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum
    end
end

uvs[:,2] ./= √(norm(umode(2))^2 - norm(vmode(2))^2)
qs = [ψ + 0.07u*umode(2) + 0.07conj(u)*vmode(2) for u in uu]
PF = @animate for j = 2:length(uu)
    plot(
        p(qs[j]),
        p(pci(qs[1:j]))
    )
end
gif(PF, "../figs/resp200828f.gif", fps=2)

uvs[:,3] ./= √(norm(umode(3))^2 - norm(vmode(3))^2)
qs = [ψ + 0.07u*umode(3) + 0.07conj(u)*vmode(3) for u in uu]
PG = @animate for j = 2:length(uu)
    plot(
        p(qs[j]),
        p(pci(qs[1:j]))
    )
end
gif(PG, "../figs/resp200828g.gif", fps=2)
