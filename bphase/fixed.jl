# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, DifferentialEquations, Interpolations

using Revise
using Superfluids

include("plotsty.jl")

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FourierDiscretisation{2}(200, 20/199)

dt = 1e-4
R = 1.5	# orbit radius for GPE plots

z = Superfluids.argand(d)
r = abs.(z)

L = Superfluids.operators(s,d, :L) |> only

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
R_TF = sqrt(2μ)

# interpolate nin
y = d.h/2*(1-d.n:2:d.n-1)
f = CubicSplineInterpolation((y, y), ψ)
yy = range(y[1], y[end]; length=5*d.n)
rs = hypot.(yy', yy)
ψ = f.(yy', yy)
ψ ./= norm(ψ)
nin = [sum(@. abs2(ψ)*(rs<s)) for s=0:d.h/5:R_TF]

rr = range(0.0, R_TF, length=20)
rr = rr[2:end]

gp = similar(rr)	# GPE
ip = similar(rr)	# imprinted
ss = similar(rr)	# end radii
nv = similar(rr)	# nin minus core
hs = similar(rr)
ws = similar(rr)

qs = similar(rr, Any)
s1 = similar(rr, Any)
s2 = similar(rr, Any)

Ω = 0.3

berry_diff(u,v) = imag(sum(conj(v).*u))

function imprint_phase(u)
    r₀ = find_vortex(d,u)
    @. abs(u)*(z-r₀)/abs(z-r₀)
end

if false

# Threads.@threads for j = eachindex(rr)
for j = eachindex(rr)
    rv = rr[j]
    @info "Thread starting" j id=Threads.threadid()
    g_tol, h = 1e-7, 0.6
    ws[j] = Ω
    qs[j] = steady_state(s, d; rvs=Complex{Float64}[rv], Ω, g_tol, iterations=5000, a=0.2)
    # ws[j], qs[j] = relax_orbit(s, d, rv; g_tol, iterations=1000)
    r₀ = find_vortex(d, qs[j])
    ss[j] = abs(r₀)
    nv[j] = sum(@. abs2(qs[j])*(r<ss[j]))
    @info "Steady state" j id=Threads.threadid()
    s1[j], s2[j] = Superfluids.integrate(s, d, qs[j], [h/2Ω, h/Ω]; μ)
    r₁ = find_vortex(d, s1[j])
    r₂ = find_vortex(d, s2[j])
    @info "Rotation" j id=Threads.threadid() theory=h/2 first=angle(conj(r₀)*r₁) last=angle(conj(r₁)*r₂)
    hs[j] = θ = angle(conj(r₀)*r₂)
    @info "GPE Berry" j first=2berry_diff(s1[j], qs[j])/h last=2berry_diff(s2[j], s1[j])/h
    gp[j] = berry_diff(s2[j], qs[j])/θ
    ip[j] = berry_diff(imprint_phase(s2[j]), imprint_phase(qs[j]))/θ
end

PF = plot()
plot!([R/R_TF, R/R_TF], [0.0, 0.4]; snapsty...)
plot!(0:d.h/5R_TF:1, nin; insty...)
scatter!(ss/R_TF, ip; impsty...)
scatter!(ss/R_TF, gp; bpsty...)
xlims!(0,1)
xticks!([0, 0.5, 1])
savefig(PF, "resp200812e.pdf")

plot(scatter(ss/R_TF, -gp), scatter(ss/R_TF, hs), layout=@layout [a;b])

end