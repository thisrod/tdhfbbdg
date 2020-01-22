# Order parameter for a Bose-Einstein condensate with a lattice of
# seven vortices

using LinearAlgebra, BandedMatrices, JLD2

C = 10;  μ=25;  Ω=2*0.55
h = 0.2;  N = 100
a = 1.7

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π)
ψ = z.^7 .*ψ./sqrt(1 .+ r²).^7
# jitter to include L ≠ 7 components
ψ += (0.1*randn(N,N) + 0.1im*randn(N,N)).*abs.(ψ)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ

residual = []
ψ₀ = similar(ψ)
for _ = 1:1
    ψ₀ .= ψ
    for k = keys(ψ)
        i,j = Tuple(k)
        ψ[k] = 0
        T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
        L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
        ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
            (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])))
        ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
     end
     Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ₀)).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
     m = sum(conj.(ψ).*Lψ)/norm(ψ)^2
     push!(residual, norm(Lψ-m*ψ)/norm(ψ))
end

