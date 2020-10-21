# relax a pair of vortices using general Superfluids code

using LinearAlgebra, Optim, JLD2

using Revise
using Superfluids
using Superfluids: find_vortices, poles, cluster_adjacent, adjacent_index

default(:legend, :none)
Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FDDiscretisation{2}(100, 0.2, 7)
g_tol = 1e-7

L, H, J = Superfluids.operators(s,d,:L,:H,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(2μ)
Ω = 0.3

H = Superfluids.operators(s,d,:H) |> only
result = optimize(0.2R_TF, 0.5R_TF, abs_tol=g_tol) do r
    q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=5000)
    real(dot(H(q;Ω),q))
end
r = result.minimizer
q = steady_state(s, d; rvs=Complex{Float64}[-r+0.2, r+0.2], Ω, g_tol, iterations=5000)

tt = 0:0.05:1
S = Superfluids.integrate(s, d, q,tt)

@save "poff.jld2" tt S

# for j in eachindex(qs); plot(d,@.qs[j]/abs(qs[j])); scatter!([rv1[j]]) |> display; sleep(0.2); end

r1s = Complex{Float64}[]
r2s = Complex{Float64}[]
for q in qs
    ra, rb = sort(find_vortices(d, q), by=real)
    push!(r1s, ra)
    push!(r2s, rb)
end