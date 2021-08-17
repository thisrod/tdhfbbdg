# Moat potential with internal rotating trap

using DifferentialEquations
using LinearAlgebra: dot
using Revise
using Superfluids
using Superfluids: cloud

# include("plotsty.jl")

R = 2.6
w = 0.2
rv = complex(1.0)

Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))

Vmoat(x,y) = (x^2+y^2)/2 + 100*exp(-(hypot(x,y)-R)^2/2/w^2)
s0 = Superfluid{2}(500, Vmoat)
Superfluids.default!(s)
d = FourierDiscretisation{2}(200, 20/199)
Superfluids.default!(d)
z = argand()
r = argand(abs)

u = cloud(d, rv)
u[@. abs(z) > R] = abs.(u[@. abs(z) > R])

# TODO adjust frame rotation frequency
q = steady_state(rvs=[rv], initial=u, a=0.1, iterations=5000)

J, L = operators(s0, d, :J, :L)
Ω, μ = [J(q)[:] q[:]] \ L(q)[:] |> real


μoff = 0.0	# NaN for lock step
dt = 10^-4.5

t(x) = (tanh(x)+1)/2
χ = @. t((R+r)/w)*t((R-r)/w)	# inner trap characteristic fn

isnan(μoff) && (μoff = Ω)

# Set chemical potential to zero, then shift inner potential
μL = dot(q, L(q)) |> real
s1 = Superfluid{2}(500, (x,y) -> Vmoat(x,y) - μL)

# solve dynamics
P = ODEProblem((ψ,_,_)->-1im*L(ψ), q, (0.0,2π/Ω))
S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.1)

Su = S.u
St = S.t

@save "/fred/oz127/rpolking/moat.jld2" Ω R w rv Su St

