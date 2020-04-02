# Berry phase for a phase singularity and an imprinted vortex

using Plots, ComplexPhasePortrait
using LinearAlgebra

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))

r₀ = 2.0
r₁ = 4.0	# image vortex
h = 0.03;  N = 300
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y

A = π*r₀^2

function ψ(θ)
    q = z .- r₀*exp(1im*θ)
    q ./= abs.(q)
end

ψr(θ) = exp(-1im*θ)*ψ(θ)

function φ(θ)
    r = r₀*exp(1im*θ)
    q = z .- r
    q ./= sqrt.(1 .+ abs2.(z .- r))
end

# two vortices
function vim(θ)
    q = z .- r₀*exp(1im*θ)
    q .*= z .- r₁*exp(1im*θ)
    q ./= abs.(q)
end

vimr(θ) = exp(-2im*θ)*vim(θ)

# vortex and image
vam(θ) = vam(θ,θ)
function vam(θ,φ)
    q = z .- r₀*exp(1im*θ)
    q .*= conj.(z .- r₁*exp(1im*φ))
    q ./= abs.(q)
end

vamr(θ) = exp(-im*θ)*vam(θ)

mask!(u) = (u .*= (abs.(z).≤N*h/2))

∫(u) = h^2*sum(u)

# The integrals of all these should all converge to 2πA
cdiff(u,θ₀,θ₁) = 1im*conj.(u(0.5(θ₀+θ₁))).*(u(θ₁)-u(θ₀))
mdiff(u,θ₀,θ₁) = 0.5im*conj.(u(θ₀)+u(θ₁)).*(u(θ₁)-u(θ₀))
ndiff(u,θ₀,θ₁) = imag.(conj.(u(θ₁)).*u(θ₀))

function ∮(u, difun, m)
    h = 2π/m
    s = zero(z)
    for j = 1:m
       s .+= difun(u, (j-1)*h, j*h)
    end
    s
end

plotlcr(L,C,R) = plot(zplot(L), zplot(C), zplot(R), layout=@layout [a b c])
plotdiff(a...) = plotlcr(cdiff(a...), mdiff(a...), ndiff(a...))

halsum() = -2π*(abs.(z) .< r₀)

plotcis(u, m) = zplot(∮(u, ndiff, m))
ploterr(u, m) = zplot(add(u, ndiff, m).-halsum())

# truncs = [∫(A.*(abs.(z).<j)) |> real for j = 1:15]

k₀ = 2π/(N*h)
dyn = zero(z)
for i = -5:5
    for j = -5:5
        dyn .+= randn(2)⋅[1, 1im]*exp.(1im*k₀*(i*x.+j*y))
    end
end
dyn = exp.(1im*real.(dyn)/10)