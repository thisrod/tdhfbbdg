# Acquire a rotating vortex orbit by bisection search

source = open(@__FILE__) do f
    read(f, String)
end

using LinearAlgebra, BandedMatrices, Optim, JLD2
using Statistics: mean

C = 3000
N = 100
l = 20.0	# maximum domain size
dts = 10 .^ (-5:-0.5:-9.0)	# residual
Ωs = [0.0, 0.6]

# r₀ = 1.7
r₀ = 2.5		# offset of imprinted vortex

h = min(l/(N+1), sqrt(√2*π/N))
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
J(ψ) = -1im*(x.*(∂*ψ)-y.*(ψ*∂'))
L(ψ,Ω) = T(ψ)+(V+U(ψ)).*ψ-Ω*J(ψ)
K(ψ) = T(ψ)+(V+U(ψ)).*ψ		# lab frame
H(ψ,Ω) = T(ψ)+(V+U(ψ)/2).*ψ-Ω*J(ψ)
E(ψ) = dot(ψ,H(ψ,Ω)) |> real
grdt!(buf,ψ) = copyto!(buf, 2*L(ψ,Ω))
togrid(xy) = reshape(xy, size(z))

while true

    φ = @. cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
    φ .*= (z.-r₀)
    φ ./= norm(φ)
    
    Ω = mean(Ωs)
    
    for r = dts
        result = optimize(
            ψ -> dot(ψ,H(ψ,Ω)) |> real,
            (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
            φ,
            ConjugateGradient(manifold=Sphere()),
            Optim.Options(iterations=10_000, g_tol=r, allow_f_increases=true)
        )
        φ .= result.minimizer
        Jφ = real(dot(φ, J(φ)))
        if Jφ < 0.1
            Ωs[1] = Ω
            break
        elseif Jφ > 0.9
            Ωs[2] = Ω
            break
        end
    end

    @save "acqorbit.jld2" source N h Ωs φ
    0.1 < real(dot(φ, J(φ))) < 0.9  &&  break
end
