# Order-parameter dynamics for an offset Bose vortex

using LinearAlgebra, BandedMatrices, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

C = 10;  μ = 12;  Ω = 2*0
h = 0.2;  N = 30
a = 1.4;  b = 0.2		# SOR and thermal polations

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
k₀ = Tuple(keys(r²)[argmin(abs.(z.-0.5))])
ψ = Complex.(exp.(-r²/2)/√π);  ψ = conj.(z).*ψ./sqrt(1 .+ r²)
# jitter to include L ≠ 0 component
ψ += (0.1*randn(N,N) + 0.1im*randn(N,N)).*abs.(ψ)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# find ground state solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ

function impose_vortex!(ψ)
    # boundary condition ψ(z[k₀]) = 0, ψ is analytic at z[k₀]
    udlr = [(1,0),(0,1),(-1,0),(0,-1)]
    ψ[CartesianIndex(k₀)] = 0
    α = sum(ψ[CartesianIndex(k₀.+l)]/(l⋅(1,1im)) for l = udlr) / 4
    for l = udlr
        ψ[CartesianIndex(k₀.+l)] = α*(l⋅(1,1im))
    end
end

# solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ

ψ₀ = similar(ψ)
nnc = zero(ψ)
potential = []
residual = []
for _ = 1:Int(1000)
    ψ₀ .= ψ
    for k = keys(ψ)
        i,j = Tuple(k)
        ψ[k] = 0
        T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
        L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
        ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
            (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
        ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
        impose_vortex!(ψ)
     end
     Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ₀)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
     # take residual of unconstrained components
     impose_vortex!(Lψ)
     E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
     push!(potential, E)
     push!(residual, norm(Lψ-E*ψ)/norm(ψ))
end

# propagate

P = ODEProblem((ψ,p,t)->-1im*(-∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))), ψ, (0.0,1.0))
S = solve(P)

# L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
