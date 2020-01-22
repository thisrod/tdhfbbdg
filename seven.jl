# Order parameter for a Bose-Einstein condensate with a lattice of
# seven vortices

using LinearAlgebra, BandedMatrices, Optim

g = 10/sqrt(2);  Ω=0.55*sqrt(2)
Nc = 116.24

h = 0.2/2^(1/4);  N = 100
C = g*Nc/h^2		# Optim sets norm(ψ) = 1

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π)
ψ = z.^7 .*ψ./sqrt(1 .+ r²).^7
# jitter to include L ≠ 7 components
ψ += (0.1*randn(N,N) + 0.1im*randn(N,N)).*abs.(ψ)
ψ ./= norm(ψ)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴-Ω·ψ*Jψ
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
H(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+C/2*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(ψ) = sum(conj.(ψ).*H(ψ)) |> real

grdt!(buf,ψ) = copyto!(buf, L(ψ))

# result = optimize(E, grdt!, ψ, ConjugateGradient(manifold=Sphere()));
# ψ₀ = result.minimizer;

function rdl(ψ)
    μ = sum(conj.(ψ).*L(ψ)) |> real
    norm(L(ψ)/μ-ψ)
end
