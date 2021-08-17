# Find limits of convergence for precessing vortex pairs

using LinearAlgebra, JLD2, Printf, Plots, Optim
using Superfluids
using Superfluids: bdg_output

R = 10
s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 2R/99) |> Superfluids.default!
L, J = Superfluids.operators(:L,:J)

Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))
Plots.default(:legend, :none)

g_tol = 1e-6

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(2μ)

rr = (0.5:0.5:4.5)/5*R_TF

function wdisc(r, w, a)
    q = steady_state(s, d; rvs=[r, -r], Ω=w, g_tol, iterations=5000, as=[a, a])
    w2, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
    w2-w
end

ws = []
qs = []

for r in rr
    rv = complex(r)
    a = 0.3 + 0.7*r/R_TF
    wend = 0.4 + 0.02*r/R_TF
    rv = complex(r)
    result = optimize(w->abs2(wdisc(rv, w, a)), 0.1, wend, abs_tol=g_tol)
    Ω = result.minimizer
    q = steady_state(s, d; rvs=[rv, -rv], Ω, g_tol, iterations=5000, as=[a, a])
    push!(ws, Ω)
    push!(qs, q)
end
