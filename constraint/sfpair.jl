# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

s = Superfluid{2}(500, (x,y)->x^2+y^2)
Superfluids.default!(s)
d = FDDiscretisation{2}(150, 20/149)
Superfluids.default!(d)
Superfluids.default!(:g_tol, 1e-7)
g_tol = 1e-7

L, H, _, _ = Superfluids.operators(s,d)

ψ = relaxed_state()
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(μ)

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

if false

W = 0.45
# rr = d.h:d.h:R_TF
rr = range(d.h, R_TF, length=15)
qs = [Superfluids.relax_field(s, d, Complex{Float64}[-r, r], W; g_tol, iterations=1000)
    for r = rr]
lEs = [real(dot(H(q),q)) for q in qs]
rEs = [real(dot(H(q,W),q)) for q in qs]
rdls = rsdl.(qs, W)

plot(
    scatter(rr/R_TF, rEs.-E₀, xshowaxis=false, ylabel="E (rot)",
        title="vortex pair, rotating frame W=0.45"),
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

# use pairplots.jl

end