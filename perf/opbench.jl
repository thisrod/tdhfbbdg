# Single vortex relaxation benchmark
#
# For compatibility with numerical libraries, this file uses the
# normalisation ∫|ψ|² = 1 and a nonlinear term gN|ψ|².  In contrast,
# the rest of this package uses ∫|ψ|² = N and g|ψ|².

# Questions for each method:
#
# 1. How many steps does it take to start geometric convergence?
#
# 2. What is the rate of geometric convergence after that?
#
# 3. How much work do the steps take?
#
# 4. How close to an eigenstate does the initial guess need to be?

using LinearAlgebra, BandedMatrices, Optim

const C = 10*20.31^2	# chosen to get norm(ψ) ≈ 1 with μ = 10
const μ = 10 
const Ω = 2*0.575
const h = 0.2
const N = 40

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

const ∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
const ∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

const y = h/2*(1-N:2:N-1)
const x = y'
const z = Complex.(x,y)
const V = const r² = abs2.(z)

# The first guess converges to 0 or 2 vortices, never 1.
# ψ = conj.(z).*exp.(-r²/2)/√π
ψ = Complex.(exp.(-r²/2)/√π);  ψ = conj.(z).*ψ./sqrt(1 .+ r²)

# solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
function time_sor(φ, steps)

    a = 1.4		# SOR polation
    
    ψ = copy(φ)
    ψ₀ = similar(ψ)
    println("Order parameter by SOR")
    n = 0;  residual = Inf
    rsdls = []
    while n < steps
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*abs2.(ψ₀[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
         n += 1
         if n % 100 == 0
             Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
             E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
             residual = norm(Lψ-E*ψ)/norm(ψ)
             push!(rsdls, residual)
         end
    end
    
    ψ, rsdls

end # time_sor

# ψ₁, rsdl = time_sor(ψ, 4000);
# ψ₁ ./= norm(ψ₁);

# Find minimum energy with Optim.  The necessary gradient is the RHS of the GPE.

L(ψ) = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
H(ψ) = -∂²*ψ-ψ*∂²+V.*ψ+C/2*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(ψ) = sum(conj.(ψ).*H(ψ)) |> real

function rdl(ψ)
    μ = sum(conj.(ψ).*L(ψ)) |> real
    norm(L(ψ)-μ*ψ)
end

cost(xy) = E(reshape(xy,N,N))
grdt!(buf,xy) = copyto!(buf, L(reshape(xy,N,N))[:])

# result = optimize(cost, grdt!, ψ[:], ConjugateGradient(manifold=Sphere()))
# ψ₂ = reshape(result.minimizer,N,N);

# wall_time = work * eachindex(rsdls) ./ length(rsdls)
# geometric convergence after 10*100 steps
# m, c = [wall_time[10:end] ones(size(wall_time[10:end]))] \ log.(10, rsdls[10:end])

# TODO try out https://github.com/JuliaMolSim/DFTK.jl/blob/master/examples/gross_pitaevskii.jl
