# Energy landscape for an orbiting vortex and 3-lattice

using LinearAlgebra, Plots, Optim, DifferentialEquations

using Revise
using Superfluids

default(:legend, :none)

s = Superfluid{2}(3000, (x,y)->x^2+y^2)
Superfluids.default!(s)
d = FDDiscretisation(100, 20)
Superfluids.default!(d)
Superfluids.default!(:g_tol, 1e-6)

L, H = Superfluids.operators(:L, :H)

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
R_TF = √μ

rr = 0:0.2:R_TF
Ωs = 0.2:0.025:0.5

function rsdl(q, Ω)
    Lq = L(q,Ω)
    μ = dot(Lq,q)
    norm(Lq-μ*q)
end

# compare energy to vortex-free steady state at Ω
oqs = Array{Any}(undef,length(rr),length(Ωs))
rEs = similar(oqs, Float64)
lEs = similar(oqs, Float64)
rdls = similar(oqs, Float64)
for j = eachindex(Ωs)
    for k = eachindex(rr)
        q = Superfluids.relax_field(s, d, [Complex(rr[k])], Ωs[j])
        oqs[k,j] = q
        lEs[k,j] = real(dot(H(q, Ωs[j]), q))
        rEs[k,j] = real(dot(H(q), q))
        rdls[k,j] = rsdl(q, Ωs[j])
    end
end

function relaxed_energy(s, d, r, Ω)
    q = Superfluids.relax_field(s, d, r, Ω)
    real(dot(H(q, Ω), q))
end

# optimize(r -> -relaxed_energy(s, d, complex(r), 0.3), 0.0, R_TF)

P = plot()
for j = eachindex(Ωs)
    plot!(rr./R_TF, lEs[:,j], label="$(Ωs[j])", leg=:none)
end
xlabel!("r_v/R_TF")
ylabel!("E")

# dynamics

q = oqs[11,5]
m = real(dot(L(q), q))
P = ODEProblem((ψ,_,_)->-1im*(L(ψ)-m*ψ), q, (0.0,1.0/0.4))
S = solve(P, RK4(), adaptive=false, dt=1e-4, saveat=0.05/0.4)

if false

rr = 0.5:0.2:R_TF
uu = exp.(2π*1im*(0:2)/3)
tqs = [Superfluids.relax_field(s, d, r*uu, Ω) for r in rr[2:end]]
tEs = [dot(H(q, Ω), q) |> real for q in tqs]
rdls = [rsdl(q, Ω) for q in tqs]

# plot(scatter(rr, oEs), scatter(rr, rdls), layout=@layout [a;b])

end