# relax an order parameter with a pair of vortices

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

Superfluids.default!(Superfluid{2}(3000, (x,y)->x^2+y^2))
Superfluids.default!(FDDiscretisation(100, 20))


z = argand()
r = abs.(z)

struct NormVorts <: Manifold
   ixs
   U
   function NormVorts(r::Number)
       d = Superfluids.default(:discretisation)
        z = argand(d)
        ixs = zeros(Int, 8)
        ixs[1:4] = sort(eachindex(z), by=j->abs(z[j]-r))[1:4] 
        ixs[5:8] = sort(eachindex(z), by=j->abs(z[j]+r))[1:4]
        os = zeros(eltype(z),8,2)
        a = normalize!(z[ixs[1:4]].-r)
        o = ones(eltype(z), 4)
        o .-= a*(a'*o)
        os[1:4,1] = normalize(o)
        a = normalize!(z[ixs[5:8]].+r)
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

L, H = Superfluids.operators(:L, :H)

relaxed_op(R, Ω, g_tol) = relax(R, Ω, g_tol).minimizer

relax(R, Ω, g_tol) =
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        normalize((z.-R).*(z.+R).*cloud()),
        ConjugateGradient(manifold=NormVorts(R)),
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
qs = [relaxed_op(1.5, w, 1e-3) for w in ws]
rdls = [rsdl(q, w) for (q, w) in zip(qs, ws)]

function p(u)
    plot(Superfluids.default(:discretisation), u)
    scatter!([-1.5, 1.5], [0, 0], mc=:white, xlims=(-5,5), ylims=(-5,5))
end

Ω = relaxed_orbit(1.7, 1e-3).minimizer
ψ = relaxed_op(1.7, Ω, 1e-3)


# Find Kelvin mode.  Need to use SM, so sparse matrices don't help

μ = sum(conj.(ψ).*L(ψ)) |> real

function op2mat(f)
    M = similar(z,length(z),length(z))
    u = similar(z)
    for j = eachindex(u)
        u .= 0
        u[j] = 1
        M[:,j] = f(u)[:]
    end
    M
end

C = Superfluids.default(:superfluid).C
h = Superfluids.default(:discretisation).h
T, V, U, J = Superfluids.operators(:T, :V, :U, :J)

BdGmat = [
    op2mat(φ->T(φ)+(V+2U(ψ)).*φ-Ω*J(φ)-μ*φ)    op2mat(φ->@. -C/h*ψ^2*φ);
    op2mat(φ->@. C/h*conj(ψ)^2*φ)    op2mat(φ->-T(φ)-(V+2U(ψ)).*φ-Ω*J(φ)+μ*φ)
];



ωs,uvs,nconv,niter,nmult,resid = eigs(BdGmat; nev=16, which=:SM) 

umode(j) = reshape(uvs[1:N^2, j], N, N)
vmode(j) = reshape(uvs[N^2+1:end, j], N, N)
