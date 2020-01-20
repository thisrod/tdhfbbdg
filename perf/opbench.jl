# Single vortex relaxation benchmark

using LinearAlgebra, BandedMatrices, Optim

# The benchmark task is to find the least-energy solution of the
# static 2D Gross-Pitaevskii equation, with a harmonic potential V =
# x² + y², in a frame rotating with angular velocity Ω:
#
# -∇²ψ/2 + (V+g*abs2.(ψ)).*ψ - 1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
#
# As Ω increases from 0, the solution will change from a vortex-free
# wave function, to a single vortex at the origin, to a vortex lattice.
# The following choice gives a single vortex.

const g = 2917.0
const Ω = 0.8132

# The wave function is discretized on a rectangular grid.  The current
# grid is too small to solve the continuous problem precisely, but
# the discrete problem is a useful size for a benchmark.

const h = 0.1682
const N = 40

const y = h/2*(1-N:2:N-1)
const x = y'
const z = Complex.(x,y)
const V = const r² = abs2.(z)

# For compatibility with numerical libraries, the normalisation is
# chosen to be ∫|ψ|² = h², so that ψ discretises to a unit vector.
# The following choice of μ is consistent with that normalisation and
# the above g and Ω.

const μ = 7.071

# The first set of solutions use a finite difference derivative
# formula, with zero boundary conditions.  The derivative operators
# is implemented by a matrix ∂, such that ∂*ψ evaluates ∂ψ/∂y, and
# ψ*∂' evaluates ∂ψ/∂x.
#
# TODO: Implement finite differences with periodic boundary conditions,
# to allow the same continuous problem to be solved with SOR, Optim
# and DFTK.

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

const ∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
const ∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# The initial guess is a single vortex, so that the solvers merely
# have to relax the radial density of the wave function.

ψ = Complex.(exp.(-r²/2)/√π)
ψ = conj.(z).*ψ./sqrt(1 .+ r²)
ψ ./= norm(ψ)

# The first solution method is successive over-relaxation, an
# extrapolated Gauss-Seidel.  This is a total kludge, but it is used
# in physics work because it tends to converge to the right answer.
# It finds a ψ for a fixed value of μ.  Here μ has been chosen in
# advance to give norm(ψ) = 1.

function time_sor(φ, steps)

    a = 1.4		# SOR polation
    
    ψ = copy(φ)
    ψ₀ = similar(ψ)
    println("Order parameter by SOR")
    for n=1:steps
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = (∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j])/2
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-∂²[1,1]+V[k]+g*abs2.(ψ₀[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
    end
    
    ψ

end # time_sor

# ψ₁ = time_sor(ψ, 4000);

# Pretty pictures
#
# using Plots, ComplexPhasePortrait
# zplot(ψ) = plot(x[:], y, 
#     portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)),
#     aspect_ratio=1)
# zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))

# The first challenger is the Optim.jl library, which finds a wave
# function that minimizes the energy under the constraint norm(ψ) =
# 1.  The solver requires a gradient of the energy function.  The
# following code minimises a cost function E/h², whose gradient is
# L(ψ), the left-hand side of the GPE.

L(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+g*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
H(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+g/2*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(ψ) = sum(conj.(ψ).*H(ψ)) |> real

cost(xy) = E(reshape(xy,N,N))
grdt!(buf,xy) = copyto!(buf, L(reshape(xy,N,N))[:])

# result = optimize(cost, grdt!, ψ[:], ConjugateGradient(manifold=Sphere()));
# ψ₂ = reshape(result.minimizer,N,N);

function rdl(ψ)
    μ = sum(conj.(ψ).*L(ψ)) |> real
    norm(L(ψ)-μ*ψ)
end

# TODO try out https://github.com/JuliaMolSim/DFTK.jl/blob/master/examples/gross_pitaevskii.jl
