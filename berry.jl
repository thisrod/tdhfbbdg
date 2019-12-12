# numerical validation of Berry phase for a singular vortex

using Plots, ComplexPhasePortrait
using LinearAlgebra

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))

r₀ = 2.0
r₁ = r₀*exp(0.5im)
h = 0.3;  N = 36
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y

A = π*r₀^2

function ψ(θ)
    q = z .- r₀*exp(1im*θ)
    q ./= abs.(q)
    mask!(q)
end

mask!(u) = (u .*= (abs.(z).≤N*h/2))

# these should all converge to A as ε→0
cdiff(ψ,ε) = 1im*h^2/ε*sum(conj.(ψ(0)).*(ψ(ε/2)-ψ(-ε/2)))
mdiff(ψ,ε) = 1im*h^2/ε*sum(0.5(ψ(ε)+ψ(-ε/2)).*(ψ(ε)-ψ(-ε/2)))
ndiff(ψ,ε) = h^2/ε*sum(imag.(conj.(ψ(ε/2)).*ψ(-ε/2)))

εs = 10 .^(-4:0.2:-1)

cs = cdiff.(ψ,εs);
@assert norm(imag.(cs),Inf) < 1e-8
cs = real.(cs);

# scatter(εs, abs.(cs.+A), mc=:black, leg=:none)