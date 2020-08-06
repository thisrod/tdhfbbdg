# Moat potential

using DifferentialEquations, JLD2

# C = 10_000.0
# N = 400
# l = 15.0
# R = 2.6
# w = 0.2
# μoff = 4
gtol = 10^-11.5
dt = 10^-4.5

@load "mu.jld2"

include("../system.jl")
include("../figs.jl")

# Add moat
@. V += 100*exp(-(r-R)^2/2/w^2)

# φ = ground_state(φ, 0, gtol)
# 
# # Set chemical potential to zero outside the moat, and shift inner potential
# 
# t(x) = (tanh(x)+1)/2
# χ = @. t((R+r)/w)*t((R-r)/w)
# μL = dot(φ, L(φ)) |> real
# @. V += μoff*χ - μL
# P = ODEProblem((ψ,_,_)->-1im*L(ψ), φ, (0.0,1.5))
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

mx = @. (R-w) < r < (R+w)
rr = r[mx]
hh = -0.2:0.0002:0.2
function Df(q,s)
    ff = angle.(q[mx])
    Dfs = Float64[]
    ix = sortperm(@. abs(rr-s))[1:250]
    cs = (@. ((rr[ix]-s))^(0:2)') \ ff[ix]
    cs[2]
end

function derplot(q)
    P1 = scatter(r[mx], angle.(q[mx]), mc=:black, ms=1.5, msw=0, leg=:none)
    xlabel!("r")
    ylabel!("arg(psi)")
    xlims!(R+hh[1], R+hh[end])
    dd = [Df(q, R+a) for a in hh]
    P2 = scatter(R.+hh, dd, mc=:black, ms=1.5, msw=0, leg=:none)
    xlabel!("r")
    ylabel!("displacement current density")
    plot(P1,P2)
end

# scatter((r[mx].-R), angle.(Su[end][mx]), mc=:green, ms=1.5, msw=0,  leg=:none)
# plot(-1:0.05:1, (-1:0.05:1).^(0:2)'*cs, leg=:none)
# plot(, scatter(rs.+R, Dfs, leg=:none))
