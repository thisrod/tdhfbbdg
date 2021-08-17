# Verify the Goldstone mode for a lattice comparable to prl-92-060407

using LinearAlgebra, JLD2, Plots
using Superfluids
using Superfluids: relax, cloud, bdg_operator, find_vortices

s = Superfluid{2}(500.0, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(200, 0.150) |> Superfluids.default!
L, J = Superfluids.operators(:L,:J)

Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))
Plots.default(:legend, :none)

p(q) = plot(plot(d,q), plot(d, @.q/abs(q)))

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
RTF = sqrt(2μ)
uu = [0.0; @. exp(2π*im*(0:6)/6)]

Ω = 0.8

q = steady_state(initial=cloud(0.5RTF*uu); Ω, iterations=10_000)
rvs = find_vortices(d, q)
u = L(q)

core!() = scatter!(real(rvs), imag(rvs), mc=:white, msc=:black, ms=3, msw=1)
core(u) = (plot(d,u);  core!())
p(q) = plot(core.([q, @.q/abs(q)])...)

B = bdg_operator(s, d, q, Ω)
u1, v1 = B(u, conj(u))