# Moat potential with internal rotating trap

using DifferentialEquations, JLD2

Ea = √2

C = 10_000.0
N = 100
l = 15.0
R = 2.6
w = 0.2
μoff = NaN	# NaN for lock step
dt = 10^-4.5

include("../system.jl")
include("../figs.jl")

# Add moat and internal stirring
@. V += 100*exp(-(r-R)^2/2/w^2)
t(x) = (tanh(x)+1)/2
χ = @. t((R+r)/w)*t((R-r)/w)	# inner trap characteristic fn
J(ψ) = -1im*χ.*(x.*(∂*ψ)-y.*(ψ*∂'))

# Ω, φ = orbit_frequency(-1.2, 3e-4, moat=true)
Ω, φ = orbit_frequency(-1.0, 3e-4, moat=true)
isnan(μoff) && (μoff = Ω)

# Set chemical potential to zero, then shift inner potential
μL = dot(φ, L(φ)) |> real
@. V += μoff*χ - μL

# solve dynamics
P = ODEProblem((ψ,_,_)->-1im*L(ψ), φ, (0.0,1.0))
S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.1)

