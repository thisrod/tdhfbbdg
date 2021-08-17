# relax an order parameter with a vortex position constraint

# ssh -l rpolking ozstar.swin.edu.au 'cd CON; sbatch runjob'

using LinearAlgebra, Optim, JLD2
using Superfluids

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)
L, J = Superfluids.operators(s,d,:L,:J)

g_tol = 1e-6

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

ws = similar(rr)
qs = similar(rr, Any)
S = similar(rr, Any)

for j = eachindex(rr)
    rv = complex(rr[j])
    a = 0.3 + 0.7*j/length(rr)
    wend = 0.4 + 0.02*j/length(rr)
    println("Relaxing ", j)
    flush(stdout)
    @time result = optimize(w->abs2(wdisc(rv, w, a)), 0.1, wend, abs_tol=g_tol)
    Ω = result.minimizer
    ws[j] = Ω
    qs[j] = steady_state(s, d; rvs=[rv], Ω, g_tol, iterations=5000, as=[a])
    tt = (1:6)*d.h/rr[j]/Ω
    println("Integrating ", j)
    flush(stdout)
    @time S[j] = Superfluids.integrate(s, d, qs[j], tt; μ)
end

# @save "fixed.jld2" rr ws qs S