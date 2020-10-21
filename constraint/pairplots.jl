default(:legend, :none)
using Superfluids: unroll

N = d.n

# Fix normalisation
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

PE = p(us[3], vs[3])
title!("green KT")
savefig("../figs/resp200828e.pdf")

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
function pci(S)
    if length(S) == 1
        zero(S[])
    else
        [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum
    end
end

qs1 = [q + 0.2u*us[2] + 0.2conj(u)*vs[2] for u in hh]
PF = @animate for j = 2:length(hh)
    P = p(qs1[j])
    rs = [rvs[k]+roff(q,0.2us[2],0.2vs[2],hh[j],k) for k = 1:2]
    scatter!(real(rs), imag(rs))
    plot!(real(rvs[1].+rv1), imag(rvs[1].+rv1), lc=:white)
    plot(P, p(pci(qs1[1:j])))
end
gif(PF, "../figs/foo.gif", fps=2)

# bp = sum(pci(qs))

qs2 = [q + 0.2u*us[3] + 0.2conj(u)*vs[3] for u in hh]
PG = @animate for j = 2:length(hh)
    P = p(qs2[j])
    rs = [rvs[k]+roff(q,0.2us[3],0.2vs[3],hh[j],k) for k = 1:2]
    scatter!(real(rs), imag(rs))
    plot!(real(rvs[2].+rv2), imag(rvs[2].+rv2), lc=:white)
    plot(P, p(pci(qs2[1:j])))
end
gif(PG, "../figs/bar.gif", fps=2)

uh = unroll(angle.(hh))
PH = scatter(uh, [sum(abs2, pci(qs1[1:j])) for j = eachindex(hh)])
scatter!(uh, [sum(abs2, (@. real(z)<0).*pci(qs1[1:j])) for j = eachindex(hh)])
scatter!(uh, [sum(abs2, (@. real(z)>0).*pci(qs1[1:j])) for j = eachindex(hh)])
plot!(uh, ncore*area2*uh)