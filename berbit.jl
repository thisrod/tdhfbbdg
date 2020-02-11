# Berry phase for an orbiting order parameter

using LinearAlgebra, BandedMatrices, Optim, JLD2

g = 10/sqrt(2);  Ω=0.15*sqrt(2)
r₀ = (1.86279296875+1.86328125)/2		# offset of imprinted phase, bisection search
Nc = 140.2749

h = 0.15/2^(1/4);  N = 200
C = g*Nc/h^2

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# iterate to self-consistent thermal cloud

ψ = z.-r₀

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴-Ω·ψ*Jψ
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
Ham(ψ) = -(∂²*ψ+ψ*∂²)/2+V.*ψ+C/2*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(xy) = sum(conj.(togrid(xy)).*Ham(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

rc = 5	# condensate radius, 
P = Diagonal(sqrt.(rc^4 .+ V[:].^2))

result = optimize(E, grdt!, ψ[:],
    GradientDescent(manifold=Sphere(), P=P),
    Optim.Options(iterations = 10_000)
)
ψ = togrid(result.minimizer)

@save "orb.jld2" ψ