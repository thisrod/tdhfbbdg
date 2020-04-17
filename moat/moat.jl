# ground state and GPE dynamics for a harmonic trap with a moat

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, Arpack
using Plots, ComplexPhasePortrait

# C = 250.0
C = 350.0
Ω = 0.0
R = 1.3
w = 0.15	# moat width
# ω = -2.86	# potential offset outside moat for lock step
# ω = -10.0	# potential offset outside moat for fast vortex
ω = 0.0

h = 0.05
N = 120
# h = 0.3
# N = 20


y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
r² = abs2.(z)

V = r²  + 20*R^2*exp.(-(r.-R).^2/2/w^2)
W = V .+ ω*(r.>R)

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

rc = 2	# condensate radius, 
P = Diagonal(sqrt.(rc^4 .+ V[:].^2))

# TODO relax residual tolerance
init = z.*(r .< (R-w))
result = optimize(E, grdt!, z[:],
    GradientDescent(manifold=Sphere()),
    Optim.Options(iterations = 10000, allow_f_increases=true)
)
ψ = togrid(result.minimizer)

φ = copy(ψ)
φ[r.>R] = abs.(φ[r.>R])

# Offset.  TODO fix parameters

kelvin = exp.(-abs2.(z)/1/R^2)
φ .+= 0.025kelvin

# Define this early for cut, paste and @load
# Offset W in place of V
f(ψ,_,_) = -1im*(-(∂²*ψ+ψ*∂²')/2+(W.-m).*ψ+C/h*abs2.(ψ).*ψ)

# Solve the GPE

Lφ = -(∂²*φ+φ*∂²')/2+V.*φ+C/2h*abs2.(φ).*φ
m = sum(conj.(φ).*Lφ) |> real

P = ODEProblem(f, φ, (0.0,1.0), saveat=0.05)
# S = solve(P)

function poles(u)
    st = [-h 0 h]
    rs = st .+ 1im*st'
    v = u ./ abs.(u)
    A = [rs.*v[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    B = [conj.(rs).*v[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    A, B
end

function locmax(u, ix::CartesianIndex)
    j, k = Tuple(ix)
    1 < j < size(u,1) && 1 < k < size(u,2) &&
        abs(u[j,k]) > 100 * minimum(abs, u) &&
        abs(u[j,k]) ≈ maximum(abs, u[j-1:j+1, k-1:k+1])
end

locmax(u) = [k for k in keys(u) if locmax(u, k)]

function show_vortices(u)
    P, Q = poles(u)
    function markup!(X, col)
        zin = z[2:end-1, 2:end-1]
        f!(R, sym) = scatter!(X, real.(zin[locmax(R)]), imag.(zin[locmax(R)]), m=sym, ms=1, mc=col, msw=0, leg=:none)
        f!(Q, :cross)
        f!(P, :xcross)
    end
    A, B = (heatmap(x[2:end-1], y[2:end-1], abs.(reverse(v,dims=1)), aspect_ratio=1) for v in poles(u))
    C = zplot(u)
    markup!(C, :white)
    D = argplot(u)
    markup!(D, :black)
    plot(C, D, B, A, layout = @layout [a b; c d])
end