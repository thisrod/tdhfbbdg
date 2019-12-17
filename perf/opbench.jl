# Single vortex relaxation benchmark

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

const C = 10
const μ = 10
const Ω = 2*0.575
const h = 0.2
const N = 40
const rdtol = 1e-5	# residual tolerance

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
             # if residual > rdtol || break
         end
    end
    
    ψ, rsdls

end # time_sor

# find minimum energy with Optim

function E(ψ)
    Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
    # ψ is normalised by constraint
    sum(conj.(ψ).*Lψ) |> real
end

# break out real and imaginary parts, constraint is unit sphere

function reconstruct(xy)
    n = length(xy) ÷ 2
    ψ = xy[1:n] + 1im*xy[n+1:end]
    m = sqrt(n) |> Int
    ψ = reshape(ψ,m,m)
end

# norm taken from SOR solution
cost(xy) = E(20.31*reconstruct(xy))

init = ψ/norm(ψ)

# result = optimize(cost, [real.(init[:]); imag.(init[:])], NelderMead(manifold=Sphere()))

# wall_time = work * eachindex(rsdls) ./ length(rsdls)
# geometric convergence after 10*100 steps
# m, c = [wall_time[10:end] ones(size(wall_time[10:end]))] \ log.(10, rsdls[10:end])

# TODO try out https://github.com/JuliaMolSim/DFTK.jl/blob/master/examples/gross_pitaevskii.jl
