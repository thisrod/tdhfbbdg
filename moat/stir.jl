# Moat potential with internal rotating trap

using DifferentialEquations, JLD2

Ea = √2

C = 10_000.0
N = 50
l = 15.0
Ω = 0.6
R = 2.6
w = 0.2
μoff = 4.0
dt = 10^-4.5

include("../system.jl")
include("../figs.jl")

# Add moat and internal stirring
@. V += 100*exp(-(r-R)^2/2/w^2)
t(x) = (tanh(x)+1)/2
χ = @. t((R+r)/w)*t((R-r)/w)	# inner trap characteristic fn
J(ψ) = -1im*χ.*(x.*(∂*ψ)-y.*(ψ*∂'))

# Imprint moat vortex to "Thomas Fermi" state
@. φ *= conj(z+R)
φ ./= norm(φ)
φ = ground_state(φ, Ω, 1e-3)
# 
# # Set chemical potential to zero, then shift inner potential
# μL = dot(φ, L(φ)) |> real
# @. V += μoff*χ - μL
# 
# # solve dynamics
# 
# function labplot(u)
#     cs = abs.(ev'*u[:])
#     ixs = cs .> 1e-20
#     ixs[1] = false
#     scatter(ew[ixs], cs[ixs], mc=:black, msw=0, ms=3, yscale=:log10, leg=:none)
# end
# 
# function labplot!(u)
#     cs = abs.(ev'*u[:])
#     ixs = cs .> 1e-20
#     ixs[1] = false
#     scatter!(ew[ixs], cs[ixs], mc=:red, msw=0, ms=3, yscale=:log10, leg=:none)
# end
# 
# function diagplot(j)
#     P1 = scatter(S.t, [norm(er(S[j])) for j = eachindex(S)],
#         ms = 3, mc=:black, msw=0, leg=:none)
#     scatter!(S.t, [abs(dot(ψ₀,S[j])) - 1 for j = eachindex(S)],
#         ms = 3, mc=:blue, msw=0, leg=:none)
#     scatter!(S.t[j:j], [norm(er(S[j]))],
#         ms = 4, mc=:red, msw=0, leg=:none)
#     P2 = zplot(er(S[j]))
#     P3 = labplot(S[j])
#     labplot!(S[1])
#     plot(P1, P2, P3, layout=@layout [a b; c])
# end
#    
# function convplot()
#     scatter(asteps,aers, xscale=:log10, yscale=:log10,
#         label="time step", leg=:bottomleft)
#     scatter!(dsteps,ders, label="residual")
#     scatter!([1.0], ders[end:end], mc=:black, label="best")
#     xlabel!("relative work")
#     ylabel!("error component")
#     title!("Convergence at t = 0.75")
# end
# 
# # T0 = dot(ψ₀, T(ψ₀))
# # U0 = dot(ψ₀, U(ψ₀).*ψ₀)
# # V0 = dot(ψ₀, V.*ψ₀)
# 
# function slice(u)
#     j = N÷2
#     sum(u[j:j+1,:], dims=1)[:]/2
# end
# 
# # scatter(y, slice(W), msw=0, mc=:black, label="V")
# # scatter!(y, 750*slice(real.(φ)).+E0, msw=0, mc=:gray, label="q")