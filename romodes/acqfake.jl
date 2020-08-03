# Berry and Wu phase for vortices at various orbits

source = open(@__FILE__) do f
    read(f, String)
end

C = 3000
N = 100
l = 20.0
dt = 1e-4

using DifferentialEquations, Interpolations
include("../system.jl")
include("../figs.jl")

ψ = ground_state(φ, 0, 1e-6)
μ = dot(L(ψ), ψ) |> real
r_TF = sqrt(μ)
yy = range(y[1], y[end]; length=5*length(y))
rs = hypot.(yy', yy)
f = CubicSplineInterpolation((y, y), ψ)
ψ = f.(yy', yy)
ψ ./= norm(ψ)
nin = [sum(@. abs2(ψ)*(rs<s)) for s=0:h/5:r_TF]

rr = r_TF*collect(0.1:0.05:0.95)
gp = Float64[]	# GPE
ip = Float64[]	# imprinted
ss = Float64[]	# end radii
nv = Float64[]	# nin minus core
for r₀ = rr
    Ω, q = orbit_frequency(r₀, 1e-3)
    μlab = dot(q, L(q)) |> real
    P = ODEProblem((ψ,_,_)->-1im*(L(ψ)-μlab*ψ), q, (0.0,0.15/Ω))
    S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.15/Ω)
    r₁ = find_vortex(S[1])
    r₂ = find_vortex(S[2])
    θ = r₂*conj(r₁) |> angle
    push!(ss, abs(r₁))
    push!(gp, berry_diff(S[1], S[2])/θ)
    push!(ip, berry_diff(imprint_phase(S[1]), imprint_phase(S[2]))/θ)
    push!(nv, sum(@. abs2(q)*(r<abs(r₁))))
end

PF = plot(0:h/5r_TF:1, nin,
    leg=:none, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=1.5,
    size=(200,200), dpi=72; insty...)
# plot!(ss/r_TF, nv, label="core")
scatter!(ss/r_TF, -ip; impsty...)
xlims!(0,1)
ylims!(0,1)
savefig(PF, "../figs/resp200714a.pdf")
