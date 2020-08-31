# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

s = Superfluid{2}(3000, (x,y)->x^2+y^2)
Superfluids.default!(s)
d = FDDiscretisation(80, 20)
Superfluids.default!(d)
Superfluids.default!(:g_tol, 1e-6)
g_tol = 1e-6

L, H = Superfluids.operators(:L, :H)

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(μ)

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

W = 0.35
rr = d.h:d.h:R_TF
qs = [Superfluids.relax_field(s, d, Complex{Float64}[-r, r], W; g_tol, iterations=1000)
    for r = rr]
lEs = [real(dot(H(q),q)) for q in qs]
rEs = [real(dot(H(q,W),q)) for q in qs]
rdls = rsdl.(qs, W)

plot(
    scatter(rr/R_TF, rEs.-E₀, xshowaxis=false, ylabel="E (rot)",
        title="vortex pair, rotating frame W=0.35"),
    scatter(rr/R_TF, rdls, xlabel="r/R_TF", ylabel="residual"),
    layout=@layout [a;b]
)

# Optimizing E gives better answer than optimizing residual
result = optimize(0.2R_TF, 0.6R_TF, abs_tol=g_tol) do r
    q = Superfluids.relax_field(s, d, Complex{Float64}[-r, r], W; g_tol, iterations=1000)
    real(dot(H(q,W),q))
end
r = result.minimizer

q = Superfluids.relax_field(s, d, Complex{Float64}[-r, r], W; g_tol, iterations=1000)

# Find Kelvin mode
# Need to use SM, so sparse matrices don't help (but BandedBlockBanded might).

ws, us, vs = Superfluids.modes(s, d, q, W, 8)

if false

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

PC = plot(p(umode(1)), p(vmode(1)))
title!("zero mode")
savefig("../figs/resp200828c.pdf")

PD = plot(p(umode(2)), p(vmode(2)))
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

end