# comparison of vortex acquisition vs orbit and constraint radius

using LinearAlgebra, Plots, DifferentialEquations, Optim

using Revise
using Superfluids
using Superfluids: relax, cloud, argand, find_vortices

Plots.default(:legend, :none)

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)
g_tol, h = 1e-7, 0.6

z = argand(d)
r = abs.(z)

J, L, H = Superfluids.operators(s,d, :J, :L, :H)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(2μ)

rr = range(0.0, R_TF, length=20)
rr = rr[2:end]

function wdisc(r, w, a)
    q = steady_state(s, d; rvs=[r], Ω=w, g_tol, iterations=5000, as=[a])
    w2, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
    w2-w
end

rv = complex(rr[end])
ww = 0.1:0.05:0.45
rts = [relax(s, d; rvs=[rv], Ω, g_tol, iterations=5000, as=[1.0]) for Ω = ww]
qs = [r.minimizer for r in rts]
ws = []
for q in qs; w, μ = [J(q)[:] q[:]] \ L(q)[:] |> real; push!(ws, w); end

result = optimize(w->abs2(wdisc(rv, w, 0.3)), ww[1], ww[end], abs_tol=g_tol)
Ω = result.minimizer

rv2 = complex(rr[end])
qs2 = [relax(s, d; initial=cloud(d, rv2), Ω, g_tol, iterations).minimizer for iterations = 1:60]
rs2 = [find_vortex(d, q) for q in qs2]
for (q, r) in zip(qs2, rs2); plot(d,q); scatter!([r]) |> display; sleep(1); end
scatter(real(rs2[4:end]))

w = steady_state(s, d; initial=q0, Ω, g_tol, iterations=1)
q = steady_state(s, d; rvs=Complex{Float64}[rv], Ω, g_tol, iterations=5000)