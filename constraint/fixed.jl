# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, Optim, DifferentialEquations, Interpolations

using Revise
using Superfluids

Plots.default(:legend, :none)

Superfluids.default!(Superfluid{2}(3000, (x,y)->x^2+y^2))
Superfluids.default!(FDDiscretisation(200, 14))
Superfluids.default!(:g_tol, 1e-3)

dt = 1e-4
R = 1.4095863217731164	# orbit radius from dummyplot

z = argand()
r = abs.(z)

struct NormVort <: Manifold
   ixs
   o
   function NormVort(r::Number)
        d = Superfluids.default(:discretisation)
        z = argand(d)
        ixs = sort(eachindex(z), by=j->abs(z[j]-r))
        ixs = ixs[1:4]
        a = normalize!(z[ixs].-r)
        o = ones(eltype(z), 4)
        o .-= a*(a'*o)
        normalize!(o)
        new(ixs, o)
   end
end

function prjct!(M, q)
    q[M.ixs] .-= M.o*(M.o'*q[M.ixs])
    q
end

# The "vortex at R" space is invariant under normalisation
Optim.retract!(M::NormVort, q) =
    Optim.retract!(Sphere(), prjct!(M, q))
Optim.project_tangent!(M::NormVort, dq, q) =
    Optim.project_tangent!(Sphere(), prjct!(M, dq),q)

L, H = Superfluids.operators(:L, :H)

relaxed_op(R, Ω, g_tol) = relax(R, Ω, g_tol).minimizer

relax(R, Ω, g_tol) =
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        normalize((z.-R).*cloud()),
        ConjugateGradient(manifold=NormVort(R)),
        Optim.Options(iterations=1000, g_tol=g_tol, allow_f_increases=true)
    )

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

# from ../romodes/acqfake.jl

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(μ)

# interpolate nin
xy = Superfluids.default(:discretisation)
y = xy.h/2*(1-xy.n:2:xy.n-1)
f = CubicSplineInterpolation((y, y), ψ)
yy = range(y[1], y[end]; length=5*xy.n)
rs = hypot.(yy', yy)
ψ = f.(yy', yy)
ψ ./= norm(ψ)
nin = [sum(@. abs2(ψ)*(rs<s)) for s=0:xy.h/5:R_TF]

rr = range(0.0, R_TF, length=20)
rr = rr[2:end]

berry_diff(u,v) = imag(sum(conj(v).*u))

function imprint_phase(u)
    r₀ = find_vortex(u)
    @. abs(u)*(z-r₀)/abs(z-r₀)
end

gp = Float64[]	# GPE
ip = Float64[]	# imprinted
ss = Float64[]	# end radii
nv = Float64[]	# nin minus core
qs = []
ws = Float64[]
hs = Float64[]

s1 = []
s2 = []

for r₀ = rr[1:length(rr)÷2]
    g_tol, h = (r₀<R_TF/4) ? (1e-7, 0.6) : (1e-5, 0.15)
    Ω = optimize(w -> rsdl(relaxed_op(r₀, w, g_tol), w), 0.0, 0.6, abs_tol=g_tol).minimizer
    @info "Steady state"
    q = relaxed_op(r₀, Ω, g_tol)
    push!(qs, q)
    push!(ws, Ω)
    P = ODEProblem((ψ,_,_)->-1im*(L(ψ)-μ*ψ), q, (0.0,h/Ω))
    S = solve(P, RK4(), adaptive=false, dt=dt, saveat=h/Ω)
    @info "Rotation"
    push!(s1, S[1])
    push!(s2, S[2])
    r₁ = find_vortex(S[1])
    r₂ = find_vortex(S[2])
    θ = r₂*conj(r₁) |> angle
    push!(hs, θ)
    push!(ss, abs(r₁))
    push!(gp, berry_diff(S[1], S[2])/θ)
    push!(ip, berry_diff(imprint_phase(S[1]), imprint_phase(S[2]))/θ)
    push!(nv, sum(@. abs2(q)*(r<abs(r₁))))
end

# Colors for Berry phase plots
snapsty = RGB(0.3,0,0)
bpsty, impsty, insty, outsty, nsty =
    distinguishable_colors(5+3, [snapsty, RGB(1,1,1), RGB(0,0,0)])[4:end]
bpsty = (ms=2, mc=bpsty, msc=0.5bpsty)
impsty = (ms=2, mc=impsty, msc=0.5impsty)
insty = (lc=insty,)
outsty = (lc=outsty,)
nsty = (lc=nsty,)
snapsty = (lc=snapsty, lw=0.5)

popts = (dpi=72, leg=:none, framestyle=:box, fontfamily="Latin Modern Sans")
imopts = (popts..., xlims=(-5,5), ylims=(-5,5), size=(200,200))
sqopts = (popts..., size=(200,200))
recopts = (popts..., size=(100,200))

PF = plot()
plot!([R/R_TF, R/R_TF], [0.0, 0.4]; snapsty..., sqopts)
plot!(0:xy.h/5R_TF:1, nin; insty...)
scatter!(ss/R_TF, -ip; impsty...)
scatter!(ss/R_TF, -gp; bpsty...)
xlims!(0,1)
xticks!([0, 0.5, 1])
# savefig(PF, "../figs/resp200812e.pdf")

# plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])