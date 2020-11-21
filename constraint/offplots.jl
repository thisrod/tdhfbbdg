# plots for pairoff.jl

using Plots, Printf

default(:legend, :none)
using Superfluids: unroll

ra, rb = sort(find_vortices(d, first(qs)), by=real)
r1s = [ra]
r2s = [rb]
for q in qs[2:end]
    rs = find_vortices(d, q)
    i1 = argmin(@. abs2(rs-r1s[end]))
    push!(r1s, rs[i1])
    i2 = argmin(@. abs2(rs-r2s[end]))
    push!(r2s, rs[i2])
end

hh = unroll(angle.(r1s))
W, c = [tt ones(size(tt))] \ hh

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
function pci(S)
    if length(S) == 1
        zero(S[])
    else
        [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum
    end
end

PA = @animate for j = eachindex(qs)[2:end]
    t, q, r1, r2 = tt[j], qs[j], r1s[j], r2s[j]
    P = plot(d,q)
    scatter!([real(r1)], [imag(r1)], mc=:white)
    scatter!([real(r2)], [imag(r2)], mc=:white)
    annotate!(4.5, 4.5, (@sprintf "%.1f" t), :white)
    plot(P, plot(d, pci(qs[1:j]))) #  |> display
    # sleep(0.2)
end
gif(PA, "../figs/resp201102a.gif", fps=5)

PB = plot(plot(r1s), plot(r2s), aspect_ratio=1)
savefig(PB, "../figs/resp201102b.pdf")

z = argand(d)
ix = argmin(@. abs(z-r))
ncore = abs2(ψ[ix])/d.h^2

# bp = sum(pci(qs))


roff = r1s.-r
uh = unroll(angle.(roff))
area1 = sum([imag(conj(roff[j-1])*roff[j]) for j in findall(uh .> -2π)[2:end]]) / 2
PH = scatter(uh, [sum(abs2, pci(qs[1:j])) for j = eachindex(qs)])
scatter!(uh, [sum(abs2, (@. real(z)<0).*pci(qs[1:j])) for j = eachindex(qs)])
# scatter!(uh, [sum(abs2, (@. real(z)>0).*pci(qs[1:j])) for j = eachindex(qs)])
# plot!(uh, ncore*area1*uh)
xlabel!("theta")
ylabel!("GPE Berry phase")
title!("2pi*nin = -4e-3")
savefig("../figs/resp201027c.pdf")