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

L = only(Superfluids.operators(:L))

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

gp = similar(rr)	# GPE
ip = similar(rr)	# imprinted
ss = similar(rr)	# end radii
nv = similar(rr)	# nin minus core
ws = similar(rr)
hs = similar(rr)

qs = similar(rr, Any)
s1 = similar(rr, Any)
s2 = similar(rr, Any)


s = Superfluids.default(:superfluid)
d = Discretisation()

# for j, r₀ = pairs(rr)
Threads.@threads for j = eachindex(rr[1:5])
    r₀ = rr[j]
    @info "Thread starting" j id=Threads.threadid()
    # TODO save at halftime and check consistency between halves
    # g_tol, h = (r₀<R_TF/4) ? (1e-7, 0.6) : (1e-5, 0.15)
    g_tol, h = 0.01, 0.01
    ws[j], qs[j] = Superfluids.relax_orbit(s, d, r₀; Ωs=[0.0, 0.6], g_tol, iterations=1000)
    @info "Steady state" j id=Threads.threadid()
    P = ODEProblem((ψ,_,_)->-1im*(L(ψ)-μ*ψ), qs[j], (0.0,h/ws[j]))
    S = solve(P, RK4(), adaptive=false; dt, saveat=h/ws[j])
    @info "Rotation" j id=Threads.threadid()
    s1[j] = S[1]
    s2[j] = S[2]
    r₁ = find_vortex(S[1])
    r₂ = find_vortex(S[2])
    hs[j] = θ = r₂*conj(r₁) |> angle
    ss[j] = abs(r₁)
    gp[j] = berry_diff(S[1], S[2])/θ
    ip[j] = berry_diff(imprint_phase(S[1]), imprint_phase(S[2]))/θ
    nv[j] = sum(@. abs2(qs[j])*(r<abs(r₁)))
end

if false

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
savefig(PF, "resp200812e.pdf")

# plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])

end