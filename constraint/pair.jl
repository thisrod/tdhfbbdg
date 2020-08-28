# relax an order parameter with a pair of vortices

using LinearAlgebra, Plots, Optim, Arpack

using Revise
using Superfluids

default(:legend, :none)

Superfluids.default!(Superfluid{2}(3000, (x,y)->x^2+y^2))
Superfluids.default!(FDDiscretisation(110, 20))

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

# qs = [relaxed_op(1.5, w, 1e-3) for w in ws]
# rdls = [rsdl(q, w) for (q, w) in zip(qs, ws)]

function p(u)
    plot(Superfluids.default(:discretisation), u)
    scatter!([-1.5, 1.5], [0, 0], mc=:white, xlims=(-5,5), ylims=(-5,5))
end

Ω = relaxed_orbit(1.7, 1e-6).minimizer
ψ = relaxed_op(1.7, Ω, 1e-6)

# Find Kelvin mode
# Need to use SM, so sparse matrices don't help (but BandedBlockBanded might).

B = Superfluids.BdGmatrix(Superfluids.default(:superfluid), Superfluids.default(:discretisation), Ω, ψ)
ws,uvs,nconv,niter,nmult,resid = eigs(B; nev=16, which=:SM) 

N = Superfluids.default(:discretisation).n
umode(j) = reshape(uvs[1:N^2, j], N, N)
vmode(j) = reshape(uvs[N^2+1:end, j], N, N)

@info "max imag frequency" iw=maximum(imag, ws)
ws = real(ws)
ixs = findall(ws .> 0)
ws = ws[ixs]
uvs = uvs[:,ixs]

J = Superfluids.operators(:J)[]
jj = [dot(J(umode(k)), umode(k)) |> real for k = eachindex(ws)]

PA = p(ψ)
savefig("../figs/resp200828a.pdf")

PB = scatter(jj, ws)
scatter!(jj[2:2], ws[2:2], mc=:yellow, msc=:royalblue, msw=2)
scatter!(jj[3:3], ws[3:3], mc=:lightgreen)
xlabel!("J for u mode")
ylabel!("w (uncertain units)")
savefig("../figs/resp200828b.pdf")

PC = plot(p(umode(1)), p(vmode(1)))
title!("zero mode")
savefig("../figs/resp200828c.pdf")

PD = plot(p(umode(2)), p(vmode(2)))
title!("yellow/blue KT")
savefig("../figs/resp200828d.pdf")

PE = plot(p(umode(3)), p(vmode(3)))
title!("green KT")
savefig("../figs/resp200828e.pdf")

uu = exp.(2π*1im*(0:0.05:1))

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
function pci(S)
    if length(S) == 1
        zero(S[])
    else
        [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum
    end
end

uvs[:,2] ./= √(norm(umode(2))^2 - norm(vmode(2))^2)
qs = [ψ + 0.07u*umode(2) + 0.07conj(u)*vmode(2) for u in uu]
PF = @animate for j = 2:length(uu)
    plot(
        p(qs[j]),
        p(pci(qs[1:j]))
    )
end
gif(PF, "../figs/resp200828f.gif", fps=2)

uvs[:,3] ./= √(norm(umode(3))^2 - norm(vmode(3))^2)
qs = [ψ + 0.07u*umode(3) + 0.07conj(u)*vmode(3) for u in uu]
PG = @animate for j = 2:length(uu)
    plot(
        p(qs[j]),
        p(pci(qs[1:j]))
    )
end
gif(PG, "../figs/resp200828g.gif", fps=2)
