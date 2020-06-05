# dummy run with a harmonic trap and static condensate

source = open("null.jl") do f
    read(f, String)
end

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, JLD2

T = 20		# integration time
l = 0.15		# time step to save psi
C = 10_000.0
Ω = 0.0

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

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

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
     Optim.Options(iterations=1000, g_tol=1e-6, allow_f_increases=true)
);
φ = togrid(result.minimizer);

ramp(t) = t > 1 ? 0.0 : 0.5 + 0.5tanh(1/3t + 1/3(t-1))

# absorb KE
f(ψ,_,t) = -(1im+0.01ramp(t))*(-(∂²*ψ+ψ*∂²')/2+(V.-m-1im*(10.0ab(3.3, 0.1))).*ψ+C/h*abs2.(ψ).*ψ)

# Solve the GPE

Lφ = -(∂²*φ+φ*∂²')/2+V.*φ+C/2h*abs2.(φ).*φ
m = sum(conj.(φ).*Lφ) |> real

jldopen("moat.jld2", "w") do file
    file["C"] = C
    file["W"] = Ω
    file["y"] = y
    file["source"] = source
    j = 1
    t = 0.0
    q = φ
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
