# BdG for an offset vortex

using LinearAlgebra, BandedMatrices, DifferentialEquations, Optim, JLD2
using Statistics: mean

C = 3000
N = 100
l = 20.0	# maximum domain size

dt = 1e-4
gtol = 1e-9
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
grdt!(buf,ψ) = copyto!(buf, 2*L(ψ,Ω))
togrid(xy) = reshape(xy, size(z))

μlab = dot(φ, K(φ)) |> real
# P = ODEProblem((ψ,_,_)->-1im*(K(ψ)-μlab*ψ), φ, (0.0,30.0))
# S1 = solve(P, RK4(), adaptive=false, dt=dt, saveat=0.5)

# find vortex-free centrifugal state
ψ = @. cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
ψ ./= norm(ψ)

Ω = mean(Ωs)

result = optimize(
    ψ -> dot(ψ,H(ψ,Ω)) |> real,
    (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
    ψ,
    ConjugateGradient(manifold=Sphere()),
    Optim.Options(iterations=10_000, g_tol=gtol, allow_f_increases=true)
)
ψ .= result.minimizer

include("figs.jl")

function ivort(θ)
    R = 1.3826022522514987
    # R = [find_vortex(S[j]) for j = eachindex(S)] |> mean
    mask = @. z-R*exp(1im*θ)
    u = @. ψ*mask/sqrt(0.25^2+abs2(mask))
    u/norm(u)
end

zz1 = [find_vortex(S1[j]) for j = eachindex(S1)]
zz2 = mean(abs.(zz1))*exp(2im*π*(0:0.01:1))
q1 = S1[11]
bp1 = [conj(S1[j+1]).*S1[j] |> imag for j = 1:10] |> sum

S1t = S1.t
S1q = [S1[j] for j = eachindex(S1)]

hend = angle(zz1[11])
q2 = ivort(hend)
bp2 = [conj(ivort(hend*(j+1)/11)).*ivort(hend*j/11) |> imag for j = 1:10] |> sum

function bphase(S, ts)
    bp = [dot(S[j+1], S[j]) |> imag for j = 1:length(S)-1] |> cumsum
    bp = [0; bp]
    zs = [find_vortex(S[j]) for j = eachindex(S)]
    nin = sum(abs2.(S[1][r .< mean(abs.(zs))]))
    bp, nin*unroll(angle.(zs).-angle(zs[1]))
end

ap1, wp1 = bphase(S1q, S1t)
ap2, wp2 = bphase([ivort(angle(zz1[j])) for j = eachindex(S1)], S1t)


@save "ss.jld2" S1t S1q zz1 zz2 q1 q2 bp1 bp2 ap1 ap2 wp1 wp2