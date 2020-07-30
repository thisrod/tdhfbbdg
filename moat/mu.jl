# Moat potential

using DifferentialEquations, JLD2

# C = 10_000.0
# N = 400
# l = 15.0
# R = 2.6
# w = 0.2
# μoff = 4
# gtol = 10^-11.5
# dt = 10^-4.5

@load "mu.jld2"

include("../system.jl")
include("../figs.jl")

# Add moat
@. V += 100*exp(-(r-R)^2/2/w^2)

φ = ground_state(φ, 0, gtol)

# Set chemical potential to zero outside the moat, and shift inner potential

t(x) = (tanh(x)+1)/2
χ = @. t((R+r)/w)*t((R-r)/w)
μL = dot(φ, L(φ)) |> real
@. V += μoff*χ - μL
# P = ODEProblem((ψ,_,_)->-1im*L(ψ), φ, (0.0,0.75))
# S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.1)
# St = S.t
# Su = S.u
# @save "mu.jld2" C N l R w μoff St Su

function sce(u, hoff=0)
   u = slice(u)
   P1 = scatter(y, abs2.(u)/h^2, mc=:black, ms=1.5, msw=0, leg=:none)
   ylabel!("|psi|^2")
   P2 = scatter(y, unroll(angle.(u)) .+ 2π*hoff, mc=:green, ms=1.5, msw=0,  leg=:none)
   ylabel!("arg(psi)")
   xlabel!("x")
   plot(P1, P2, layout=@layout [a;b])
   P1, P2, plot(P1, P2, layout=@layout [a;b])
end

# scatter(r[:], angle.(S[end])[:])