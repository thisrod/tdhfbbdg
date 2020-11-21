# Moat potential with internal rotating trap

using Revise
using Superfluids
using Superfluids: argand, cloud, operators

include("plotsty.jl")

R = 2.6
w = 0.2
rv = complex(1.0)

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2 + 100*exp(-(hypot(x,y)-R)^2/2/w^2))
d = FourierDiscretisation{2}(200, 20/199)
z = argand(d)
r = abs(z)

u = cloud(d, rv)
u[@. abs(z) > R] = abs.(u[@. abs(z) > R])

# TODO adjust frame rotation frequency
q = steady_state(s,d; rvs=[rv], initial=u, a=0.1, iterations=5000)

J, L = operators(s,d, :J, :L)
Ω, μ = [J(q)[:] q[:]] \ L(q)[:] |> real


μoff = 0.0	# NaN for lock step
dt = 10^-4.5

t(x) = (tanh(x)+1)/2
χ = @. t((R+r)/w)*t((R-r)/w)	# inner trap characteristic fn

isnan(μoff) && (μoff = Ω)

# Set chemical potential to zero, then shift inner potential
μL = dot(φ, L(φ)) |> real
@. V += μoff*χ - μL

# solve dynamics
P = ODEProblem((ψ,_,_)->-1im*L(ψ), φ, (0.0,2π/Ω))
S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.1)

Su = S.u
St = S.t

@save "/fred/oz127/rpolking/moc.jld2" source Ω C N l R w Su St

