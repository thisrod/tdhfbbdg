# ground state and GPE dynamics for a harmonic trap with a moat

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, JLD2

T = 20		# integration time
l = 0.15		# time step to save psi
C = 10_000.0
Ω = 0.0
R = 1.7
# w = 0.1
w = 0.2	# double moat width
ω = -3.0

h = 0.05
N = 150

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
r² = abs2.(z)

# Absorbing boundary

function ab(rvac, bord)
    out = @. exp(-(r-rvac)^2/2/bord^2)
    out[r .> rvac] .= 1
    out
end

mt = @. exp(-(r-R)^2/2/w^2)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂ = (1/h).*op([0, -1/2, 0, 1/2, 0])
∂² = (1/h^2).*op(Float64[0, 0, 1, -2, 1, 0, 0])

# kludge BC
∂[1,:] .= ∂[2,:]
∂[end,:] .= ∂[end-1,:]
∂²[1,:] .= ∂²[2,:]
∂²[end,:] .= ∂²[end-1,:]

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴-Ω·ψ*Jψ
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
Ham(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/2h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(xy) = sum(conj.(togrid(xy)).*Ham(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

# relax density at boundary

V = r² + 100*ab(3, 0.2)

φ = similar(z);
fill!(φ,1);
result = optimize(E, grdt!, φ[:],
     GradientDescent(manifold=Sphere()),
     Optim.Options(iterations=3, g_tol=1e-6, allow_f_increases=true)
);
φ = togrid(result.minimizer);

V += 20*R^2*mt
W = V .+ ω*(r.>R)

# relax soliton phase to moat vortex

phs = z .+ 0.7
@. phs /= abs(phs)
φ[@. abs(z) < R] .*= phs[@. abs(z) < R]

# relax high momenta

a = 0.01	# width of Gaussian to convolve
φ1 = similar(φ);
for j = 1:N
    for k = 1:N
        φ1[j,k] = sum(@. φ*exp(-abs2(z-z[j,k])/2/a))
    end
end

# relax moat density to residual where vortex disappears
ψ = copy(φ1);
result = optimize(E, grdt!, ψ[:],
     GradientDescent(manifold=Sphere()),
     Optim.Options(iterations=100, g_tol=0.02, allow_f_increases=true)
 );
ψ = togrid(result.minimizer);

# Offset W in place of V, absorb KE
f(ψ,_,_) = -1im*(-(∂²*ψ+ψ*∂²')/2+(W.-m-1im*(10.0ab(3.3, 0.1))).*ψ+C/h*abs2.(ψ).*ψ)

# Solve the GPE

Lψ = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/2h*abs2.(ψ).*ψ
m = sum(conj.(ψ).*Lψ) |> real

jldopen("moat.jld2", "w") do file
    file["C"] = C
    file["W"] = Ω
    file["R"] = R
    file["y"] = y
    file["w"] = w
    j = 1
    t = 0.0
    q = ψ
    file["t0"] = t
    file["psi0"] = q
    while t ≤ T
        S = ODEProblem(f, q, (t, t+l), saveat=l) |> solve
        t = S.t[end]
        q = S[end]
        file["t$(j)"] = t
        file["psi$(j)"] = q
        S.retcode == :Success || break
        j  += 1
    end
    file["steps"] = j - 1
end
