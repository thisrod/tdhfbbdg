# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

Superfluids.default!(Superfluid{2}(3000, (x,y)->x^2+y^2))
Superfluids.default!(FDDiscretisation(80, 20))


z = argand()
r = abs.(z)

L, H = Superfluids.operators(:L, :H)

relaxed_op(R, Ω, g_tol) = relax(R, Ω, g_tol).minimizer

relax(R, Ω, g_tol) =
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        normalize((z.-R).*(z.+R).*cloud()),
        ConjugateGradient(manifold=PinnedVortices(-R, R)),
        Optim.Options(iterations=1000, g_tol=g_tol, allow_f_increases=true)
    )

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

relaxed_orbit(R, g_tol) =
    optimize(w->rsdl(relaxed_op(R, w, g_tol), w), 0.0, 0.6)

ws = 0:0.1:1
# qs = [relaxed_op(1.5, w, 1e-3) for w in ws]
# rdls = [rsdl(q, w) for (q, w) in zip(qs, ws)]

function p(u)
    plot(Superfluids.default(:discretisation), u)
    scatter!([-1.5, 1.5], [0, 0], mc=:white, xlims=(-5,5), ylims=(-5,5))
end

# Ω = relaxed_orbit(1.7, 1e-3).minimizer
# ψ = relaxed_op(1.7, Ω, 1e-3)

