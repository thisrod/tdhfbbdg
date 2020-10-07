using LinearAlgebra, Plots, Optim

using Revise
using Superfluids
using Superfluids: winding, cloud
using Superfluids: find_vortices, poles, cluster_adjacent, adjacent_index

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FDDiscretisation{2}(200, 20/199)

L, J = Superfluids.operators(s,d,:L,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(2μ)

lix = Superfluids.loopixs(d, 1.1R_TF)

rr = R_TF*(0:0.1:1)
qq = [steady_state(s,d; rvs=[complex(r)]) for r in rr]
ww = [winding(q, lix) for q in qq]
rdls = Float64[]
for q in qq
    Ω, μ = [J(q)[:] q[:]] \ L(q)[:]
    push!(rdls, sum(abs2, L(q;Ω)-μ*q))
end

p(qs...) = plot([plot(d,q) for q in qs]...)

j=5

q0 = cloud(d, complex(rr[j]))
Superfluids.relax(s,d; rvs=[complex(rr[j])], relaxer=GradientDescent, initial=q0, iterations=3)
qv = ans.minimizer

Superfluids.relax(s,d; relaxer=GradientDescent, initial=q0, iterations=3)
qn = ans.minimizer