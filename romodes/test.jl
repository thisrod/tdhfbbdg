# BdG for an offset vortex, with numerical improvements from ../basics

using LinearAlgebra, BandedMatrices, Arpack, Optim, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

C = 2748.85
N = 72
h = 0.178		# equipartition h = 0.178, but close enough
Ω = 0.288153076171875

r₀ = 1.9		# offset of imprinted phase

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op(Float64[-1/2, 0, 1/2])
∂² = (1/h^2).*op(Float64[1, -2, 1])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴
#
# The GPE functional L(ψ) is the gradient required by Optim.

T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
J(ψ) = -1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
L(ψ) = T(ψ)+(V+U(ψ)).*ψ+J(ψ)
K(ψ) = T(ψ)+(V+U(ψ)).*ψ		# lab frame
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ+J(ψ)
E(xy) = sum(conj.(togrid(xy)).*H(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

# starting point for relaxation, check boundary smoothness
φ = Complex.(exp.(-r²/2))
φ .*= (z.-r₀)
φ ./= norm(φ)

    # expand u over self-consistent eigenstates in rotating frame
    Hmat = similar(z, N^2, N^2)
    v = similar(z)
    for j = 1:N^2
        v .= 0
        v[j] = 1
        Hmat[:,j] = (T(v)+(V+U(φ)).*v+J(v))[:]
    end
    ew, ev = eigen(Hmat)
    cs = abs.(ev'*φ[:])
    ixs = cs .> 1e-20
#    scatter(ew[ixs], cs[ixs], mc=:black, msw=0, ms=3, yscale=:log10, leg=:none)

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))
