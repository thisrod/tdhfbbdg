# relax an order parameter with a pair of vortices

using LinearAlgebra, Plots, Optim

using Revise
using Superfluids

default(:legend, :none)

gas = Superfluid{2}(3000, (x,y)->x^2+y^2)
xy = FDDiscretisation(gas, 100, 20)

z = argand(xy)
r = abs.(z)

struct NormVorts <: Manifold
   ixs
   U
   function NormVorts(d::Discretisation, r::Number)
        z = argand(d)
        ixs = zeros(Int, 8)
        ixs[1:4] = sort(eachindex(z), by=j->abs(z[j]-r))[1:4] 
        ixs[5:8] = sort(eachindex(z), by=j->abs(z[j]+r))[1:4]
        os = zeros(8,2)
        a = normalize!(ixs[1:4].-r)
        o = ones(eltype(z), 4)
        o .-= a*(a'*o)
        os[1:4,1] = normalize(o)
        a = normalize!(ixs[5:8].+r)
        o = ones(eltype(z), 4)
        o .-= a*(a'*o)
        os[5:8,2] = normalize(o)
        new(ixs, os)
   end
end

function prjct!(M, q)
    q[M.ixs] .-= M.U*(M.U'*q[M.ixs])
    q
end

# The "vortex at R" space is invariant under normalisation
Optim.retract!(M::NormVorts, q) =
    Optim.retract!(Sphere(), prjct!(M, q))
Optim.project_tangent!(M::NormVorts, dq, q) =
    Optim.project_tangent!(Sphere(), prjct!(M, dq),q)

_, _, L, H = Superfluids.operators(gas, xy)

relaxed_op(R, Ω, g_tol) = relax(R, Ω, g_tol).minimizer

relax(R, Ω, g_tol) =
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        (z.-R).*(z.+R).*cloud(xy),
        GradientDescent(manifold=NormVorts(xy, R)),
        Optim.Options(iterations=1000, g_tol=g_tol, allow_f_increases=true)
    )

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

relaxed_orbit(R, g_tol) =
    optimize(w->rsdl(relaxed_op(R, w, g_tol), w), 0.0, 0.6)