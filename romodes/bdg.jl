# BdG for an offset vortex

using LinearAlgebra, BandedMatrices, DifferentialEquations, Optim, JLD2
using Statistics: mean

C = 3000
N = 100
l = 20.0	# maximum domain size

dt = 1e-4
h = min(l/(N+1), sqrt(√2*π/N))
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op(Float64[-1/2, 0, 1/2])
∂² = (1/h^2).*op(Float64[1, -2, 1])

# starting point for relaxation
@load "acqorbit.jld2" φ Ωs

T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
J(ψ) = -1im*(x.*(∂*ψ)-y.*(ψ*∂'))
L(ψ,Ω) = T(ψ)+(V+U(ψ)).*ψ-Ω*J(ψ)
K(ψ) = T(ψ)+(V+U(ψ)).*ψ		# lab frame
H(ψ,Ω) = T(ψ)+(V+U(ψ)/2).*ψ-Ω*J(ψ)
E(ψ) = dot(ψ,H(ψ,Ω)) |> real


# Solve dynamics

μlab = dot(ψ, K(ψ)) |> real
# P = ODEProblem((ψ,_,_)->-1im*(K(ψ)-μlab*ψ), φ, (0.0,30.0))
# S1 = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.5)
P = ODEProblem((ψ,_,_)->-1im*(K(ψ)-μlab*ψ), φ, (0.0,0.15))
S1 = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.01)

include("figs.jl")

zz1 = [find_vortex(S1[j]) for j = eachindex(S1)]
R = mean(abs.(zz1))

function ivort(θ)
    mask = @. z-R*exp(1im*θ)
    u = @. ψ*mask/sqrt(0.25^2+abs2(mask))
    u/norm(u)
end

zz2 = R*exp.(2im*π*(0:0.01:1))
q1 = S1[11]
bp1 = [conj(S1[j+1]).*S1[j] |> imag for j = 1:10] |> sum
nin1 = sum(abs2.(φ[r .< R]))

S1t = S1.t
S1q = [S1[j] for j = eachindex(S1)]

hend = angle(zz1[11])
q2 = ivort(hend)
bp2 = [conj(ivort(hend*(j+1)/11)).*ivort(hend*j/11) |> imag for j = 1:10] |> sum
nin2 = sum(abs2.(ψ[r .< R]))

function bphase(S)
    bp = [dot(S[j+1], S[j]) |> imag for j = 1:length(S)-1] |> cumsum
    [0; bp]
end

ap1 = bphase(S1q)
wp1 = nin1*unroll(angle.(zz1).-angle(zz1[1]))
ap2 = bphase([ivort(angle(zz1[j])) for j = eachindex(S1)])
wp2 = nin2*unroll(angle.(zz1).-angle(zz1[1]))

# @save "ss.jld2" Ω S1t S1q zz1 zz2 q1 q2 bp1 bp2 ap1 ap2 wp1 wp2 nin1 nin2