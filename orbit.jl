# orbiting dynamics for a relaxed vortex in the lab frame

using LinearAlgebra, BandedMatrices, JLD2, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

C = 10;  μ=30;  Ω=2*0.15
r₀ = 0.5		# offset of imprinted phase
# C = 10;  μ=30;  Ω=2*0.13
# r₀ = 0.2		# offset of imprinted phase
h = 0.3;  N = 36
a = 1.4		# SOR polation

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π);  ψ = (z.-r₀).*ψ

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# components of BdG matrix

eye = Matrix(I,N,N)
H = -kron(eye, ∂²) - kron(∂², eye) + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))

# iterate to self-consistent thermal cloud

nnc = zero(ψ)
ψ₀ = similar(ψ)
therm = []
oprm = []
rsdls = []
μs = []

memoized = false
memfile = "ofst.jld2"
if isfile(memfile)
    @load memfile ψ prms
    memoized =
        size(ψ) == (N,N) &&
        prms == Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
end

if memoized
    println("Order parameter: here's one I prepared earlier")
else

    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
    
    rsdl = []
    for n = 0:5000
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
         Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
         m = sum(conj.(ψ).*Lψ)/norm(ψ)^2
         push!(μs, m)
         push!(rsdl, norm(Lψ-m*ψ)/norm(ψ))
         n % 50 == 0 && push!(oprm, copy(ψ₀))
    end
    
    prms = Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
    @save memfile ψ prms
end

#    nnc += b*(reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ)) - nnc)
#    push!(oprm, copy(ψ₀))
#    push!(therm, nnc)

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ
m = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real

P = ODEProblem((ψ,_,_)->-1im*(-∂²*ψ-ψ*∂²+(V.-m).*ψ+C*abs2.(ψ).*ψ), ψ, (0.0,25.0))
S = solve(P)

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
