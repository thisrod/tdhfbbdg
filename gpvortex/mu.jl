# Moat potential

using DifferentialEquations, JLD2
using LinearAlgebra: dot
using Superfluids
using Superfluids: cloud, operators

R = 2.6
w = 0.2
μoff = 4
gtol = 10^-11.5
dt = 10^-4.5

Vmoat(x,y) = (x^2+y^2)/2 + 100*exp(-(hypot(x,y)-R)^2/2/w^2)
s0 = Superfluid{2}(500, Vmoat)

d = FourierDiscretisation{2}(400, 15/399)
Superfluids.default!(d)

L0 = operators(s0, d, :L) |> only
q = steady_state(s0)
μL = dot(q, L0(q)) |> real

# Set chemical potential to zero outside the moat, and shift inner potential

t(x) = (tanh(x)+1)/2
χ(x,y) = 
    let r = hypot(x,y)
        t((R+r)/w)*t((R-r)/w)
    end

s1 = Superfluid{2}(500, (x,y) -> Vmoat(x,y) + μoff*χ(x,y) - μL)
L = operators(s1, d, :L) |> only
P = ODEProblem((ψ,_,_)->-1im*L(ψ), q, (0.0,1.5))
S = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.1)
St = S.t
Su = S.u
# @save "/fred/oz127/rpolking/mu.jld2" R w μoff St Su
